"""
Product Integration — Test File 3
PI-07, PI-08, PI-09

Run from the project root:

python3 eval/product_integration/test_product_physical_integration.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png
"""

import argparse

from product_testing_utils import run_multi_criterion_test, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical product integration tests")
    parser.add_argument("product", help="Original uploaded product image")
    parser.add_argument("category", help="Expected product category")
    parser.add_argument("--output", required=True, help="Generated advertisement PNG")
    args = parser.parse_args()

    result = run_multi_criterion_test(
        "PI-07/PI-08/PI-09",
        "Product-Body Fit, Anatomical Interaction & Occlusion",
        args.product,
        args.output,
        args.category,
        [
            (
                "body_fit_attachment",
                "Does the product look physically worn, held, carried or attached "
                "to the person in a believable way, rather than floating nearby?",
            ),
            (
                "anatomical_interaction",
                "Does the product interact naturally with relevant body parts such "
                "as hands, fingers, face, ears, neck or wrist, without obvious "
                "anatomical errors?",
            ),
            (
                "occlusion_layering",
                "Are overlaps between the product, body and clothing physically "
                "believable, with no obvious product-through-body or incorrect "
                "front/behind layering?",
            ),
        ],
    )
    save_result(result)


if __name__ == "__main__":
    main()
