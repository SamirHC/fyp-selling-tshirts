from datetime import datetime
from src.design_generation import generate


if __name__ == "__main__":
    import os
    import webbrowser

    design = generate.generate_random_design_from_db()

    out_path = os.path.join("out", f"main {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.svg")
    with open(out_path, "w") as f:
        f.write(design.to_svg())
    webbrowser.get("firefox").open(out_path)
