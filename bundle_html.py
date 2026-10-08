import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(curr_dir, "uncertain_grid_restoration", "dashboard")
html_path = os.path.join(base_dir, "index.html")
css_path = os.path.join(base_dir, "style.css")
js_path = os.path.join(base_dir, "app.js")

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

with open(css_path, "r", encoding="utf-8") as f:
    css = f.read()

with open(js_path, "r", encoding="utf-8") as f:
    js = f.read()

# Replace css link with inline style
html_standalone = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}\n</style>")

# Replace script tag with inline script
html_standalone = html_standalone.replace('<script src="app.js"></script>', f"<script>\n{js}\n</script>")

out_path1 = os.path.join(curr_dir, "dashboard.html")

with open(out_path1, "w", encoding="utf-8") as f:
    f.write(html_standalone)

print("Bundled standalone HTML file generated successfully:")
print("->", out_path1)
