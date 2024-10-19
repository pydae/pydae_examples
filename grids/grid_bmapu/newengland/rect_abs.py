import xml.etree.ElementTree as ET

def apply_transform(matrix, x, y):
    # Parse the transformation matrix (matrix="a b c d e f")
    a, b, c, d, e, f = [float(value) for value in matrix.split()[1:7]]

    # Calculate the new x and y values after applying the transformation matrix
    new_x = a * x + c * y + e
    new_y = b * x + d * y + f

    return new_x, new_y

def make_rects_absolute(svg_file):
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Find all rect elements
    for rect in root.findall(".//rect"):
        # Get the x, y, and transform attributes
        x = float(rect.get("x"))
        y = float(rect.get("y"))
        transform = rect.get("transform")

        # If a transform attribute exists, apply it to calculate new x and y
        if transform:
            new_x, new_y = apply_transform(transform, x, y)
            rect.set("x", str(new_x))
            rect.set("y", str(new_y))

    # Save the modified SVG to a new file
    modified_svg_file = "modified_" + svg_file
    tree.write(modified_svg_file)

    print(f"Modified SVG saved as {modified_svg_file}")

if __name__ == "__main__":
    svg_file = "newengland.svg"  # Replace with your SVG file's path
    make_rects_absolute(svg_file)