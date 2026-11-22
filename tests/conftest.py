# The product-integration file is a standalone script (python tests/test_product_integration.py),
# written by teammates; pytest would mis-read its helper as a fixture-based test.
collect_ignore = ["test_product_integration.py"]
