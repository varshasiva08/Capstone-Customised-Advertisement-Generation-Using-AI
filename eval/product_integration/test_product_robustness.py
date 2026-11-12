"""
Product Integration — Test File 5
PI-13, PI-14, PI-15

PI-13: Overall Product Naturalness
PI-14: Product Robustness across generated outputs
PI-15: Repeated Generation & Product Consistency

This file expects the SAME original product and multiple generated outputs.

Example:

python3 eval/product_integration/test_product_robustness.py \
  "Products to be Integrated/handbag.jpg" handbag \
  outputs/output_seed42.png \
  outputs/output_seed43.png \
  outputs/output_seed44.png \
  outputs/output_seed45.png \
  outputs/output_seed46.png
"""

import argparse
from statistics import mean

from product_testing_utils import run_multi_criterion_test, save_batch_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Product robustness tests")
    parser.add_argument("product", help="ONE original product image used for all runs")
    parser.add_argument("category", help="Expected product category")
    parser.add_argument(
        "outputs",
        nargs="+",
        help="Two or more generated advertisement PNGs for the same product",
    )
    args = parser.parse_args()

    if len(args.outputs) < 2:
        parser.error("Provide at least two generated outputs for robustness testing.")

    all_results = []

    for index, output_path in enumerate(args.outputs, start=1):
        result = run_multi_criterion_test(
            f"PI-13-G{index}",
            f"Overall Naturalness — Generation {index}",
            args.product,
            output_path,
            args.category,
            [
                (
                    "overall_naturalness",
                    "Does the product look naturally integrated into the generated "
                    "advertisement, as if it genuinely belonged in the original scene "
                    "rather than being artificially added?",
                ),
            ],
        )
        all_results.append(result)

    naturalness_scores = [
        item["scores"]["overall_naturalness"]
        for item in all_results
    ]
    successful = sum(score >= 3 for score in naturalness_scores)
    success_rate = successful / len(naturalness_scores) * 100
    average_score = mean(naturalness_scores)

    # PI-14: robustness means the product remains acceptably integrated across
    # the supplied generations.
    robustness_status = "PASS" if success_rate >= 80 else "FAIL"

    # PI-15: repeated-generation consistency is reported from the same batch.
    consistency_status = "PASS" if average_score >= 3 and success_rate >= 80 else "FAIL"

    print("\n" + "=" * 72)
    print("PI-14  Product Robustness")
    print("=" * 72)
    print(f"Generations tested : {len(naturalness_scores)}")
    print(f"Successful outputs : {successful}/{len(naturalness_scores)}")
    print(f"Success rate       : {success_rate:.1f}%")
    print(f"PI-14 RESULT       : {robustness_status}")

    print("\n" + "=" * 72)
    print("PI-15  Repeated Generation & Product Consistency")
    print("=" * 72)
    print(f"Average naturalness: {average_score:.2f}/5")
    print(f"Consistency result : {consistency_status}")

    batch = {
        "test_id": "PI-13/PI-14/PI-15",
        "category": args.category,
        "original_product": args.product,
        "generated_outputs": args.outputs,
        "pi13_generation_results": all_results,
        "pi14_success_rate": round(success_rate, 2),
        "pi14_status": robustness_status,
        "pi15_average_score": round(average_score, 2),
        "pi15_status": consistency_status,
    }
    save_batch_result("PI-13_PI-14_PI-15", batch)


if __name__ == "__main__":
    main()
