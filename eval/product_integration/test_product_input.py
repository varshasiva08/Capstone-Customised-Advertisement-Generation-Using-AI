"""
Product Integration — Test File 1
PI-01, PI-02, PI-03

Run from the project root:

python3 eval/product_integration/test_product_input.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png
"""

import argparse
from pathlib import Path

from product_testing_utils import (
    print_input_test,
    run_multi_criterion_test,
    save_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Product input tests")
    parser.add_argument("product", help="Original uploaded product image")
    parser.add_argument("category", help="Expected product category")
    parser.add_argument("--output", required=True, help="Generated advertisement PNG")
    args = parser.parse_args()

    # PI-01: the image supplied to the test is the same image that was
    # dynamically uploaded through the live application.
    result1 = print_input_test(args.product, args.category)
    save_result(result1)

    # PI-02 + PI-03: compare original product against generated advertisement.
    result23 = run_multi_criterion_test(
        "PI-02/PI-03",
        "Product Recognition & Attribute Fidelity",
        args.product,
        args.output,
        args.category,
        [
            (
                "product_identity",
                "Is the product in IMAGE 2 recognizably the same product type as "
                "the product in IMAGE 1?",
            ),
            (
                "attribute_fidelity",
                "Are the important visible attributes of the original product "
                "(especially colour, material, shape and distinctive design/features) "
                "reasonably preserved in the generated product?",
            ),
        ],
    )
    save_result(result23)


if __name__ == "__main__":
    main()
