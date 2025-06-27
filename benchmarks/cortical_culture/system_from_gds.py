import os
import hashlib
import urllib.request
import urllib.parse
import gdspy
import json
import sys


def parse_electrode_positions(source):

    if source.startswith("http"):
        parsed_url = urllib.parse.urlparse(source)
        filename = os.path.basename(parsed_url.path)

        url_hash = hashlib.md5(source.encode()).hexdigest()

        gds_filepath = os.path.join(os.path.dirname(__file__), f"{url_hash}_{filename}")

        if not os.path.exists(gds_filepath):
            print(f"Downloading {filename} from {source}...")
            try:
                urllib.request.urlretrieve(source, gds_filepath)
                print(f"Downloaded to {gds_filepath}")
            except Exception as e:
                raise RuntimeError(f"Failed to download") from e

    else:
        gds_filepath = source

    gdsii = gdspy.GdsLibrary(infile=gds_filepath)

    electrodes = []
    for cell_name in gdsii.cells:
        cell = gdsii.cells[cell_name]

        for polygon in cell.polygons:

            # detect electrode circle of 30 um
            points = polygon.polygons[0]
            if len(points) >= 8:
                center_x = points[:, 0].mean()
                center_y = points[:, 1].mean()
                center = (round(center_x), round(center_y))

                distances = (
                    (points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2
                ) ** 0.5
                avg_radius = distances.mean()

                if abs(avg_radius - (30 / 2.0)) <= 1 and distances.std() <= 1:
                    if center not in electrodes:
                        electrodes.append(center)

    return electrodes


if __name__ == "__main__":

    electrodes = parse_electrode_positions(
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://gazzolalab.github.io/MiV-OH/_downloads/3cea71bc4e589c70b679fa091398fe86/MEA128_rec.GDS"
    )

    data = {"electrodes": electrodes}

    output_path = os.path.join(os.path.dirname(__file__), "system.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(electrodes)} electrode positions to {output_path}")
