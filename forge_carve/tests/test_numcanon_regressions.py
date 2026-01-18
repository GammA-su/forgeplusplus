import numcanon

def test_json_equal_tolerates_float_epsilon_near_int():
    assert numcanon.json_equal(41610.00000000001, 41610)

def test_canon_json_intifies_near_integer():
    assert numcanon.canon_json(41610.00000000001) == 41610
