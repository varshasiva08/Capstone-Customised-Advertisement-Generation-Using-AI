# Product Integration Evaluation Suite

Place these six files under:

eval/product_integration/

Files:
- test_product_input.py
- test_product_fidelity.py
- test_product_physical_integration.py
- test_product_visual_integration.py
- test_product_robustness.py
- product_testing_utils.py

The five test files contain the 15 Product Integration test cases.
product_testing_utils.py is only a shared helper.

The tests compare an original product image with the corresponding generated
advertisement. The original image is only a local test reference; it does not
need to be part of the application.

The visual evaluator uses the same Gemma-3-27B model and Hugging Face token
rotation approach used by modules/product_describe.py.

Before running:
1. Ensure your .env contains HF_TOKEN_1 (and optionally HF_TOKEN_2/3).
2. Install project requirements.
3. Run commands from the project root.

Current product/output pairs:
- sunglasses.jpeg -> output_seed42.png
- jewel.jpeg -> output_seed43.png
- watch.jpg -> output_seed44.png
- handbag.jpg -> output_seed45.png

TEST FILE 1
-----------
python3 eval/product_integration/test_product_input.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png

TEST FILE 2
-----------
python3 eval/product_integration/test_product_fidelity.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png

TEST FILE 3
-----------
python3 eval/product_integration/test_product_physical_integration.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png

TEST FILE 4
-----------
python3 eval/product_integration/test_product_visual_integration.py \
  "Products to be Integrated/sunglasses.jpeg" sunglasses \
  --output outputs/output_seed42.png

Repeat the same four commands for:
- jewel.jpeg / jewelry / output_seed43.png
- watch.jpg / watch / output_seed44.png
- handbag.jpg / handbag / output_seed45.png

TEST FILE 5
-----------
Use the SAME product with multiple generations, for example:
python3 eval/product_integration/test_product_robustness.py \
  "Products to be Integrated/handbag.jpg" handbag \
  outputs/output_seed42.png \
  outputs/output_seed43.png \
  outputs/output_seed44.png \
  outputs/output_seed45.png \
  outputs/output_seed46.png

Note: PI-15 is a repeated-generation test and therefore should only be run
when the supplied outputs were generated from the same original product.
