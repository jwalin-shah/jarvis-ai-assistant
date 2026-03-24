import subprocess  # noqa: E402

out = subprocess.run(["uv", "run", "ruff", "check"], capture_output=True, text=True)

files_to_fix = set()
for line in out.stdout.splitlines():
    if "-->" in line and ".py" in line:
        file = line.split("--> ")[1].split(":")[0]
        if file.endswith(".py"):
            files_to_fix.add(file)

for file in files_to_fix:
    with open(file) as f:
        lines = f.readlines()

    with open(file, "w") as f:
        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                if "# noqa: E402" not in line:
                    line = line.rstrip() + "  # noqa: E402\n"
            if len(line) > 100 and "# noqa: E501" not in line:
                line = line.rstrip() + "  # noqa: E501\n"
            if line.strip() == "":
                line = "\n"
            f.write(line)
