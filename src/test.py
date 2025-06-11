import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# 🔧 Step 0: Helper function to build rich employee descriptions
def build_detailed_employee_text(emp):
    text_parts = [
        f"Employee Name: {emp['name']} (ID: {emp['empID']})",
        f"Email: {emp['mailID']}",
        f"Company: {emp['company']}",
        f"Job Level: {emp['jobLevel']}",
        "\nSkills:"
    ]

    for skill in emp.get("skills", []):
        text_parts.append(
            f"- Skill: {skill['skill']['path']} | "
            f"Proficiency: {skill.get('proficiency', 'UNKNOWN')} | "
            f"Primary: {skill['isPrimary']} | Current: {skill['isCurrent']} | "
            f"Experience: {skill['experienceProjectMths']} months"
        )

    text_parts.append("\nCourses Completed:")
    for course in emp.get("courses", []):
        text_parts.append(
            f"- {course['course']['courseName']} (Completed on {course['completedOn']})"
        )

    text_parts.append("\nCertifications:")
    for cert in emp.get("certifications", []):
        text_parts.append(
            f"- {cert['certification']['certificationName']} (Certified on {cert['certifiedOn']})"
        )

    return "\n".join(text_parts)

# 📂 Step 1: Load employee data
with open("test.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)["data"]

docs = []
ids = []
metadatas = []

for entry in raw_data:
    emp = entry["employee"]
    emp_id = emp["empID"]
    
    rich_text = build_detailed_employee_text(emp)
    docs.append(rich_text)
    ids.append(emp_id)

    metadatas.append({
        "name": emp["name"],
        "jobLevel": emp["jobLevel"],
        "mailID": emp["mailID"]
    })

# 🧠 Step 2: Vectorize
print("🔄 Encoding employee profiles into vector embeddings...")
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("microsoft/codebert-base")

embeddings = model.encode(docs).tolist()

# 🧠 Step 3: Store in ChromaDB
print("📦 Storing data into ChromaDB...")
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("employee_profiles")

# Optional: clear old data
existing_ids = collection.get()["ids"]
if existing_ids:
    collection.delete(existing_ids=existing_ids)

collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
print("✅ Data stored successfully in vector DB.")

# 🔍 Step 4: Query
while True:
    query = input("\n🔍 Enter your query (e.g., Who knows PySpark and Azure?):\n> ")
    query_embedding = model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    print("\n📊 Top Matches:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"\n🔹 Match {i+1}:\n{doc}")
