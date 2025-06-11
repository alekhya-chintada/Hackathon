import argparse
import os
import shutil

# Paths
# Get the absolute path of the current file (e.g., build.py)
current_file_path = os.path.abspath(__file__)

    # Get the project root directory (assuming build.py is in scripts/)
project_root = os.path.dirname(os.path.dirname(current_file_path))

print("Project root:", project_root)
src_path = os.path.join(project_root, "src")
vue_app_path = os.path.join(project_root, 'webapps', 'talent-pro')
dist_path = os.path.join(vue_app_path, 'dist')


def build():
    print("Building Vue app...")
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)
    os.chdir(vue_app_path)
    os.system("npm run build")

    # Copy the source folder to the destination
    shutil.copytree(dist_path, os.path.join(src_path, "dist"), dirs_exist_ok=True)
    print("Build finished.")


def start():
    print("Starting...")
    os.chdir(src_path)
    os.system("python main.py")


def start_ui_dev_server():
    os.chdir(vue_app_path)
    os.system("npm run dev")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask-Vue App Command Tool")

    # Accept 'build' or 'b' as positional argument
    parser.add_argument(
        "command",
        choices=["build", "b", "start", "s", "ui"],
        help="Command to run (build/b or start/s)"
    )

    args = parser.parse_args()

    if args.command in ["build", "b"]:
        build()
    elif args.command in ["start", "s"]:
        start()
    elif args.command in ["ui"]:
        start_ui_dev_server()
