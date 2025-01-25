import json
import os
from image_result import ImageResult


RESULTS_JSON_PATH = os.path.join('..', 'output', 'results.json')
OUTPUT_HTML_FILE = 'output.html'
GALLERY_TEMPLATE_FILE = 'output_template.html'

def generate_gallery(results: list[ImageResult]):
    image_rows = ""
    for result in results:
        row_html = f'''
            <tr class="image-row">
                <td>
                    {result.id}
                </td>
                <td>
                    <img src="{result.original_path}" alt="Original" onclick="openModal('{result.original_path}')">
                </td>
                <td>
                    <img src="{result.roberts_path}" alt="Transformed roberts" onclick="openModal('{result.roberts_path}')">
                </td>
                <td>
                    <img src="{result.prewitt_path}" alt="Transformed prewitt" onclick="openModal('{result.prewitt_path}')">
                </td>
                <td>
                    <img src="{result.sobel_path}" alt="Transformed sobel" onclick="openModal('{result.sobel_path}')">
                </td>
                <td>
                    <img src="{result.robinson_path}" alt="Transformed robinson" onclick="openModal('{result.robinson_path}')">
                </td>
                <td>
                    <img src="{result.laplace_path}" alt="Transformed laplace" onclick="openModal('{result.laplace_path}')">
                </td>
                <td>
                    <img src="{result.canny_path}" alt="Transformed canny" onclick="openModal('{result.canny_path}')">
                </td>
            </tr>'''

        image_rows += row_html
    return image_rows


def calculate_average_times(results: list[ImageResult]) -> dict:
    average_results = {
        'time_roberts': sum([r.time_roberts for r in results]) / len(results),
        'time_prewitt': sum([r.time_prewitt for r in results]) / len(results),
        'time_sobel': sum([r.time_sobel for r in results]) / len(results),
        'time_robinson': sum([r.time_robinson for r in results]) / len(results),
        'time_laplace': sum([r.time_laplace for r in results]) / len(results),
        'time_canny': sum([r.time_canny for r in results]) / len(results)
    }

    return average_results

def generate_results_html(results: list[ImageResult]):
    with open(GALLERY_TEMPLATE_FILE, 'r', encoding="utf-8") as template_file:
        html_template = template_file.read()

    gallery = generate_gallery(results)
    html_content = html_template.replace("{{gallery}}", gallery)

    avg_times = calculate_average_times(results)
    html_content = html_content.replace("{{time_roberts}}", str(avg_times['time_roberts']))
    html_content = html_content.replace("{{time_prewitt}}", str(avg_times['time_prewitt']))
    html_content = html_content.replace("{{time_sobel}}", str(avg_times['time_sobel']))
    html_content = html_content.replace("{{time_robinson}}", str(avg_times['time_robinson']))
    html_content = html_content.replace("{{time_laplace}}", str(avg_times['time_laplace']))
    html_content = html_content.replace("{{time_canny}}", str(avg_times['time_canny']))

    with open(OUTPUT_HTML_FILE, "w") as file:
        file.write(html_content)


def load_results(json_file: str) -> list[ImageResult]:
    with open(json_file, 'r') as file:
        data = json.load(file)
    return [ImageResult.from_dict(item) for item in data]


def save_results(results: list[ImageResult], json_file: str):
    with open(json_file, 'w') as file:
        json.dump([result.to_dict() for result in results], file, indent=4)


if __name__ == "__main__":
    results: list[ImageResult] = load_results(RESULTS_JSON_PATH)
    generate_results_html(results)
