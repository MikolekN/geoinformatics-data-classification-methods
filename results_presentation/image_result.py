import os
from dataclasses import asdict, dataclass
from PIL import Image


@dataclass
class ImageResult:
    id: int
    original_path: str
    is_high_resolution: bool
    is_ai_generated: bool
    is_gauss_noise: bool
    is_salt_and_pepper_noise: bool
    roberts_path: str
    prewitt_path: str
    sobel_path: str
    robinson_path: str
    laplace_path: str
    canny_path: str
    time_roberts: int
    time_prewitt: int
    time_sobel: int
    time_robinson: int
    time_laplace: int
    time_canny: int
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict) -> "ImageResult":
        image = Image.open(data["original_path"])
        width, height = image.size
        return cls(
            id=data["id"],
            original_path=data["original_path"].replace(os.sep, "/"),
            is_high_resolution=data["is_high_resolution"],
            is_ai_generated=data["is_ai_generated"],
            is_gauss_noise=data["is_gauss_noise"],
            is_salt_and_pepper_noise=data["is_salt_and_pepper_noise"],
            roberts_path=data["roberts_path"].replace(os.sep, "/"),
            prewitt_path=data["prewitt_path"].replace(os.sep, "/"),
            sobel_path=data["sobel_path"].replace(os.sep, "/"),
            robinson_path=data["robinson_path"].replace(os.sep, "/"),
            laplace_path=data["laplace_path"].replace(os.sep, "/"),
            canny_path=data["canny_path"].replace(os.sep, "/"),
            time_roberts=data["time_roberts"],
            time_prewitt=data["time_prewitt"],
            time_sobel=data["time_sobel"],
            time_robinson=data["time_robinson"],
            time_laplace=data["time_laplace"],
            time_canny=data["time_canny"],
            width=width,
            height=height
        )

    def to_dict(self) -> dict:
        return asdict(self)