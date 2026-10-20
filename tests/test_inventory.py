import pytest
from src.inventory import calculate_inventory_policy

def test_order_quantity_zero_when_stock_above_target():
    result = calculate_inventory_policy(10, 2, 3, 7, 1.65, 500)
    assert result["recommended_order_qty"] == 0

def test_order_quantity_positive_when_stock_is_zero():
    result = calculate_inventory_policy(10, 2, 3, 7, 1.65, 0)
    assert result["recommended_order_qty"] > 0
    assert result["target_stock"] >= result["reorder_point"]

def test_invalid_lead_time_rejected():
    with pytest.raises(ValueError):
        calculate_inventory_policy(10, 2, 0)

def test_negative_demand_rejected():
    with pytest.raises(ValueError):
        calculate_inventory_policy(-1, 2, 3)
