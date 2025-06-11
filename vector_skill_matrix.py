import json
import glob
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def normalize(text):
    if not text:
        return ""
    return text.strip().lower().replace("-", "").replace("_", "")

def find_skill_matches(skill_phrase, structured_data, partial=False):
    skill_phrase_norm = normalize(skill_phrase)
    skill_parts = [normalize(part) for part in skill_phrase.split('|')]
    matches = []
    for profile in structured_data:
        for s in profile["skills_struct"]:
            name = normalize(s["name"])
            if skill_phrase_norm == name:
                matches.append({"profile": profile, "skill": s})
                break
            elif partial and any(part in name for part in skill_parts):
                matches.append({"profile": profile, "skill": s})
                break
    return matches

def find_and_matches(skill_parts, structured_data):
    matches = []
    for profile in structured_data:
        for s in profile["skills_struct"]:
            name = normalize(s["name"])
            if all(part in name for part in skill_parts):
                matches.append({"profile": profile, "skill": s})
                break
    return matches

def find_or_matches(skill_parts, structured_data):
    matches = []
    for profile in structured_data:
        for s in profile["skills_struct"]:
            name = normalize(s["name"])
            if any(part in name for part in skill_parts):
                matches.append({"profile": profile, "skill": s})
                break
    return matches

def find_course_matches(skill_phrase, structured_data):
    skill_phrase_norm = normalize(skill_phrase)
    matches = []
    for profile in structured_data:
        for c in profile["courses_struct"]:
            course_name = normalize(c["name"])
            if skill_phrase_norm in course_name or any(part in course_name for part in skill_phrase_norm.split("|")):
                matches.append({"profile": profile, "course": c})
                break
    return matches

def skill_sort_key(match):
    proficiency_order = {"EXPERT": 4, "PROFICIENT": 3, "COMPETENT": 2, "BEGINNER": 1, None: 0}
    s = match["skill"]
    return (
        -(s.get("experienceProjectMths", 0) or 0),
        -1 if (s.get("isCurrent", "").upper() == "YES") else 0,
        1 if (s.get("isPrimary", "").upper() == "YES") else 0,
        -proficiency_order.get((s.get("proficiency") or "").upper(), 0)
    )

def clean_skills(skills_struct):
    seen = set()
    cleaned = []
    for s in skills_struct:
        name = normalize(s["name"])
        if name and name not in seen:
            seen.add(name)
            cleaned.append(s)
    return cleaned

# Load Data
all_data = []
for file in glob.glob("datasets/*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "data" in data:
            all_data.extend(data["data"])

raw_data = all_data

# Prepare data for embedding
model = SentenceTransformer('./all-MiniLM-L6-v2')
docs, ids, metadatas, structured_data = [], [], [], []

for entry in raw_data:
    emp = entry["employee"]
    emp_id = emp.get("empID", "")
    name = emp.get("name", "")
    job_level = emp.get("jobLevel", "")
    company = emp.get("company", "")
    mailID = emp.get("mailID", "")

    skills_struct = [
        {
            "name": s.get("skill", {}).get("path", ""),
            "proficiency": s.get("proficiency", ""),
            "isPrimary": s.get("isPrimary", ""),
            "isCurrent": s.get("isCurrent", ""),
            "experienceProjectMths": s.get("experienceProjectMths", 0) or 0
        }
        for s in emp.get("skills", []) if s.get("skill", {}).get("path")
    ]
    skills_struct = clean_skills(skills_struct)
    skills_list = [s["name"] for s in skills_struct]
    skills = ", ".join(skills_list)

    courses_struct = [
        {
            "name": c.get("course", {}).get("courseName", ""),
            "completedOn": c.get("completedon", "")
        }
        for c in emp.get("courses", []) if c.get("course", {}).get("courseName")
    ]
    courses_list = [c["name"] for c in courses_struct]
    courses = ", ".join(courses_list)

    certs_struct = [
        {
            "name": cert.get("certification", {}).get("certificationName", ""),
            "certifiedOn": cert.get("certifiedon", "")
        }
        for cert in emp.get("certifications", []) if cert.get("certification", {}).get("certificationName")
    ]
    certs_list = [c["name"] for c in certs_struct]
    certifications = ", ".join(certs_list)

    summary = f"{name} (ID: {emp_id}) is at Job Level {job_level} in {company}. Skills: {skills}. Courses: {courses}. Certifications: {certifications}."
    docs.append(summary)
    ids.append(emp_id)
    metadatas.append({
        "empID": emp_id,
        "name": name,
        "jobLevel": job_level,
        "company": company,
        "mailID": mailID,
        "skills": skills,
        "skills_list": ", ".join(skills_list),
        "courses": courses,
        "courses_list": ", ".join(courses_list),
        "certifications": certifications,
        "certs_list": ", ".join(certs_list)
    })
    structured_data.append({
        "empID": emp_id,
        "name": name,
        "jobLevel": job_level,
        "company": company,
        "mailID": mailID,
        "skills_struct": skills_struct,
        "courses_struct": courses_struct,
        "certs_struct": certs_struct,
        "summary": summary
    })

print("Encoding summaries into vector embeddings...")
embeddings = model.encode(docs, show_progress_bar=True, convert_to_tensor=True).tolist()

print("Storing data into ChromaDB...")
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="employee_profiles")
existing = collection.get()
existing_ids = existing['ids'] if 'ids' in existing else []
if existing_ids:
    collection.delete(ids=existing_ids)
collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
print("Data stored successfully in ChromaDB.")

query = input("\nEnter your query (e.g., who knows Big Data and NLP?):\n>")

def extract_skill_phrase(query):
    match = re.search(r"who knows (.+?)(\?|$)", query.lower())
    if match:
        return match.group(1).strip()
    return None

skill_phrase = extract_skill_phrase(query)
if not skill_phrase:
    print("Could not extract skill phrase from query.")
    exit()

skill_phrase = skill_phrase.replace(" and ", "|").replace(" or ", "|")
skill_parts = [normalize(part) for part in skill_phrase.split("|")]

print(f"Skill phrase for matching: {skill_phrase}")

# Main Matching Logic
top_matches = []

exact_matches = find_skill_matches(skill_phrase, structured_data)
for m in exact_matches:
    m["match_type"] = "exact"
top_matches.extend(exact_matches)

if len(skill_parts) > 1:
    and_matches = find_and_matches(skill_parts, structured_data)
    for m in and_matches:
        m["match_type"] = "and"
    existing_ids = {m["profile"]["empID"] for m in top_matches}
    and_matches = [m for m in and_matches if m["profile"]["empID"] not in existing_ids]
    top_matches.extend(and_matches)

if len(skill_parts) > 1:
    or_matches = find_or_matches(skill_parts, structured_data)
    for m in or_matches:
        m["match_type"] = "or"
    existing_ids = {m["profile"]["empID"] for m in top_matches}
    or_matches = [m for m in or_matches if m["profile"]["empID"] not in existing_ids]
    top_matches.extend(or_matches)

related_matches = find_skill_matches(skill_phrase, structured_data, partial=True)
for m in related_matches:
    m["match_type"] = "partial"
existing_ids = {m["profile"]["empID"] for m in top_matches}
related_matches = [m for m in related_matches if m["profile"]["empID"] not in existing_ids]
top_matches.extend(related_matches)

course_matches = find_course_matches(skill_phrase, structured_data)
for m in course_matches:
    m["match_type"] = "course"
existing_ids = {m["profile"]["empID"] for m in top_matches}
course_matches = [m for m in course_matches if m["profile"]["empID"] not in existing_ids]
top_matches.extend(course_matches)

if len(top_matches) < 3:
    print("\nNo direct or related skill/course match found. Showing semantic fallback:")
    query_embedding = model.encode(query, convert_to_tensor=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    existing_ids = {m["profile"]["empID"] for m in top_matches if "profile" in m}
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        if meta.get('empID', '') in existing_ids:
            continue
        print(f"\nMatch {len(top_matches)+1} (Semantic Match):")
        print(f"Name: {meta.get('name', '')} (ID: {meta.get('empID', '')}), Job Level: {meta.get('jobLevel', '')}, Company: {meta.get('company', '')}")
        print(f"Email: {meta.get('mailID', '')}")
        skills = meta.get('skills_list', '').split(', ')
        print("Top skills:")
        for skill in skills[:5]:
            print(f"- {skill}")
        if len(skills) > 5:
            print(f"...and {len(skills)-5} more.")
        if len(top_matches) + 1 >= 3:
            break

if top_matches:
    exacts = [m for m in top_matches if m.get("match_type") == "exact"]
    best_exact = sorted(exacts, key=skill_sort_key)[0] if exacts else None

    best_and = None
    if not best_exact:
        ands = [m for m in top_matches if m.get("match_type") == "and"]
        best_and = sorted(ands, key=skill_sort_key)[0] if ands else None

    non_exacts = [m for m in top_matches if m.get("match_type") not in ("exact", "and")]
    non_exacts_sorted = sorted(non_exacts, key=skill_sort_key)

    final_matches = []
    if best_exact:
        final_matches.append(best_exact)
    elif best_and:
        final_matches.append(best_and)
    final_matches.extend(non_exacts_sorted[:2])

    print("\nTop 3 relevant employees:")
    for m in final_matches:
        profile = m["profile"]
        print(f"\n{profile['name']} (ID: {profile['empID']}), Email: {profile['mailID']}")
        if "skill" in m:
            s = m["skill"]
            print(f"Skill: {s['name']}, Experience: {s.get('experienceProjectMths', 0)} months, Proficiency: {s.get('proficiency', '')}, Current: {s.get('isCurrent', '')}, Primary: {s.get('isPrimary', '')}")
        elif "course" in m:
            c = m["course"]
            print(f"Related course: {c['name']} (Completed On: {c.get('completedOn', '')})")
