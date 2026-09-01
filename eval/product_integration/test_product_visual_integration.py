"""
Product Integration — Test File 4
PI-10, PI-11, PI-12

Run from the project root:

python3 eval/product_integration/test_product_visual_integration.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png
"""

import argparse

from product_testing_utils import run_multi_criterion_test, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual product integration tests")
    parser.add_argument("product", help="Original uploaded product image")
    parser.add_argument("category", help="Expected product category")
    parser.add_argument("--output", required=True, help="Generated advertisement PNG")
    args = parser.parse_args()

    result = run_multi_criterion_test(
        "PI-10/PI-11/PI-12",
        "Perspective, Clothing Interaction & Lighting/Shadow",
        args.product,
        args.output,
        args.category,
        [
            (
                "perspective_orientation",
                "Does the product follow the person's pose, orientation and camera "
                "perspective so it appears to exist in the same scene?",
            ),
            (
                "clothing_interaction",
                "Does the product interact naturally with clothing or hair where "
                "relevant, such as a handbag strap over clothing or jewellery over "
                "clothing/hair?",
            ),
            (
                "lighting_shadow",
                "Are the product's lighting, highlights and shadows consistent "
                "with the surrounding person and scene, without looking pasted or "
                "floating?",
            ),
        ],
    )
    save_result(result)


if __name__ == "__main__":
    main()
