import json
import os

import cv2

from image_result import ImageResult
import numpy as np
from skimage.metrics import structural_similarity as ssim

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


def generate_average_times_per_image_type(results: list[ImageResult]) -> dict:
    avg_times_per_type = {
        'original': {'time_roberts': 0, 'time_prewitt': 0, 'time_sobel': 0, 'time_robinson': 0, 'time_laplace': 0,
                     'time_canny': 0},
        'gauss_noise': {'time_roberts': 0, 'time_prewitt': 0, 'time_sobel': 0, 'time_robinson': 0, 'time_laplace': 0,
                        'time_canny': 0},
        'salt_and_pepper_noise': {'time_roberts': 0, 'time_prewitt': 0, 'time_sobel': 0, 'time_robinson': 0,
                                  'time_laplace': 0, 'time_canny': 0},
        'low_resolution': {'time_roberts': 0, 'time_prewitt': 0, 'time_sobel': 0, 'time_robinson': 0, 'time_laplace': 0,
                           'time_canny': 0},
    }

    for result in results:
        if result.is_gauss_noise:
            avg_times_per_type['gauss_noise']['time_roberts'] += result.time_roberts
            avg_times_per_type['gauss_noise']['time_prewitt'] += result.time_prewitt
            avg_times_per_type['gauss_noise']['time_sobel'] += result.time_sobel
            avg_times_per_type['gauss_noise']['time_robinson'] += result.time_robinson
            avg_times_per_type['gauss_noise']['time_laplace'] += result.time_laplace
            avg_times_per_type['gauss_noise']['time_canny'] += result.time_canny
        elif result.is_salt_and_pepper_noise:
            avg_times_per_type['salt_and_pepper_noise']['time_roberts'] += result.time_roberts
            avg_times_per_type['salt_and_pepper_noise']['time_prewitt'] += result.time_prewitt
            avg_times_per_type['salt_and_pepper_noise']['time_sobel'] += result.time_sobel
            avg_times_per_type['salt_and_pepper_noise']['time_robinson'] += result.time_robinson
            avg_times_per_type['salt_and_pepper_noise']['time_laplace'] += result.time_laplace
            avg_times_per_type['salt_and_pepper_noise']['time_canny'] += result.time_canny
        elif result.is_high_resolution:
            avg_times_per_type['original']['time_roberts'] += result.time_roberts
            avg_times_per_type['original']['time_prewitt'] += result.time_prewitt
            avg_times_per_type['original']['time_sobel'] += result.time_sobel
            avg_times_per_type['original']['time_robinson'] += result.time_robinson
            avg_times_per_type['original']['time_laplace'] += result.time_laplace
            avg_times_per_type['original']['time_canny'] += result.time_canny
        elif not result.is_high_resolution:
            avg_times_per_type['low_resolution']['time_roberts'] += result.time_roberts
            avg_times_per_type['low_resolution']['time_prewitt'] += result.time_prewitt
            avg_times_per_type['low_resolution']['time_sobel'] += result.time_sobel
            avg_times_per_type['low_resolution']['time_robinson'] += result.time_robinson
            avg_times_per_type['low_resolution']['time_laplace'] += result.time_laplace
            avg_times_per_type['low_resolution']['time_canny'] += result.time_canny

    num_results = len(results)
    for image_type, times in avg_times_per_type.items():
        for key in times:
            times[key] /= num_results

    return avg_times_per_type

def generate_chart_html(times: dict) -> str:
    chart_data = {
        "labels": ['Roberts', 'Prewitt', 'Sobel', 'Robinson', 'Laplace', 'Canny'],  # Algorytmy
        "datasets": [
            {
                "label": 'Oryginalne zdjęcia',
                "data": [
                    times['original']['time_roberts'],
                    times['original']['time_prewitt'],
                    times['original']['time_sobel'],
                    times['original']['time_robinson'],
                    times['original']['time_laplace'],
                    times['original']['time_canny']
                ],
                "backgroundColor": 'rgba(54, 162, 235, 0.2)',
                "borderColor": 'rgba(54, 162, 235, 1)',
                "borderWidth": 1
            },
            {
                "label": 'Zdjęcia z szumem Gaussa',
                "data": [
                    times['gauss_noise']['time_roberts'],
                    times['gauss_noise']['time_prewitt'],
                    times['gauss_noise']['time_sobel'],
                    times['gauss_noise']['time_robinson'],
                    times['gauss_noise']['time_laplace'],
                    times['gauss_noise']['time_canny']
                ],
                "backgroundColor": 'rgba(255, 99, 132, 0.2)',
                "borderColor": 'rgba(255, 99, 132, 1)',
                "borderWidth": 1
            },
            {
                "label": 'Zdjęcia z szumem solnym i pieprzowym',
                "data": [
                    times['salt_and_pepper_noise']['time_roberts'],
                    times['salt_and_pepper_noise']['time_prewitt'],
                    times['salt_and_pepper_noise']['time_sobel'],
                    times['salt_and_pepper_noise']['time_robinson'],
                    times['salt_and_pepper_noise']['time_laplace'],
                    times['salt_and_pepper_noise']['time_canny']
                ],
                "backgroundColor": 'rgba(75, 192, 192, 0.2)',
                "borderColor": 'rgba(75, 192, 192, 1)',
                "borderWidth": 1
            },
            {
                "label": 'Zdjęcia o niskiej rozdzielczości',
                "data": [
                    times['low_resolution']['time_roberts'],
                    times['low_resolution']['time_prewitt'],
                    times['low_resolution']['time_sobel'],
                    times['low_resolution']['time_robinson'],
                    times['low_resolution']['time_laplace'],
                    times['low_resolution']['time_canny']
                ],
                "backgroundColor": 'rgba(153, 102, 255, 0.2)',
                "borderColor": 'rgba(153, 102, 255, 1)',
                "borderWidth": 1
            }
        ]
    }

    chart_html = f"""
    <div style="width: 80%; margin: 0 auto;">
        <canvas id="chart_{times.get('type', 'chart')}"></canvas>
    </div>

    <script>
        var ctx = document.getElementById('chart_{times.get('type', 'chart')}').getContext('2d');
        var chart = new Chart(ctx, {{
            type: 'bar',
            data: {chart_data},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value, index, values) {{
                                return value.toFixed(5);
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(tooltipItem) {{
                                return tooltipItem.raw.toFixed(5) + " sec";
                            }}
                        }}
                    }}
                }},
                responsive: true,
                scales: {{
                    x: {{
                        stacked: true
                    }},
                    y: {{
                        stacked: true
                    }}
                }}
            }}
        }});
    </script>
    """

    return chart_html

def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2, max_pixel=255):
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return 100
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

def calculate_ssim(image1, image2):
    return ssim(image1, image2)


def generate_comparison_results(results):
    comparison_results = {
        "roberts": {"mse": [], "psnr": [], "ssim": []},
        "prewitt": {"mse": [], "psnr": [], "ssim": []},
        "sobel": {"mse": [], "psnr": [], "ssim": []},
        "robinson": {"mse": [], "psnr": [], "ssim": []},
        "laplace": {"mse": [], "psnr": [], "ssim": []}
    }

    for result in results:
        canny_image = load_image(result.canny_path)

        for algorithm in comparison_results.keys():
            algorithm_image = load_image(getattr(result, f"{algorithm}_path"))

            mse_value = mse(algorithm_image, canny_image)
            psnr_value = psnr(algorithm_image, canny_image)
            ssim_value = calculate_ssim(algorithm_image, canny_image)

            comparison_results[algorithm]["mse"].append(mse_value)
            comparison_results[algorithm]["psnr"].append(psnr_value)
            comparison_results[algorithm]["ssim"].append(ssim_value)

    avg_comparison_results = {}
    for algorithm, values in comparison_results.items():
        avg_comparison_results[algorithm] = {
            "mse": np.mean(values["mse"]),
            "psnr": np.mean(values["psnr"]),
            "ssim": np.mean(values["ssim"])
        }

    return avg_comparison_results

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def generate_comparison_html(comparison_results):
    html_content = """
    <h2>Porównanie algorytmów z Canny</h2>
    <table class="comparison">
        <thead>
            <tr>
                <th class="header">Algorytm</th>
                <th class="header">MSE</th>
                <th class="header">PSNR</th>
                <th class="header">SSIM</th>
            </tr>
        </thead>
        <tbody>
    """

    for algorithm, results in comparison_results.items():
        html_content += f"""
        <tr>
            <td>{algorithm.capitalize()}</td>
            <td class="mse">{results['mse']:.5f}</td>
            <td class="psnr">{results['psnr']:.5f}</td>
            <td class="ssim">{results['ssim']:.5f}</td>
        </tr>
        """

    html_content += """
        </tbody>
    </table>
    """

    return html_content

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

    avg_times_per_type = generate_average_times_per_image_type(results)
    html_content = html_content.replace("{{chart_all}}", generate_chart_html(avg_times_per_type))

    COMPARISON = True
    if COMPARISON:
        comparison_results = generate_comparison_results(results)
        comparison_html = generate_comparison_html(comparison_results)
        html_content = html_content.replace("{{comparison}}", comparison_html)

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
