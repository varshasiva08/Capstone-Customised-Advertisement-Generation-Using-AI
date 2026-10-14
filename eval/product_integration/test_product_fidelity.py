"""
Product Integration — Test File 2
PI-04, PI-05, PI-06

Run from the project root:

python3 eval/product_integration/test_product_fidelity.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png
"""

import argparse

from product_testing_utils import run_multi_criterion_test, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Product fidelity tests")
    parser.add_argument("product", help="Original uploaded product image")
    parser.add_argument("category", help="Expected product category")
    parser.add_argument("--output", required=True, help="Generated advertisement PNG")
    args = parser.parse_args()

    result = run_multi_criterion_test(
        "PI-04/PI-05/PI-06",
        "Product Presence, Positioning & Scale",
        args.product,
        args.output,
        args.category,
        [
            (
                "product_presence",
                "Is the uploaded product clearly present and recognizable in the "
                "generated advertisement rather than missing?",
            ),
            (
                "product_positioning",
                "Is the product placed on/in the correct location for its category "
                "(for example, sunglasses on the face, watch on the wrist, jewellery "
                "on the appropriate body area, handbag on/near the shoulder or hand)?",
            ),
            (
                "product_scale",
                "Is the generated product realistically sized relative to the "
                "person and the scene?",
            ),
        ],
    )
    save_result(result)


if __name__ == "__main__":
    main()
