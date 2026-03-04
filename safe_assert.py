from numpy.ma.testutils import assert_array_equal, fail_if_array_equal, assert_array_almost_equal

def safe_assert_array_equal(x, y, err_msg=""):
    try:
        assert_array_equal(x, y)
    except AssertionError as e:
        details = str(e).strip()
        if err_msg:
            details = f"{err_msg}\n{details}"
        raise AssertionError(details) from None

def safe_fail_if_array_equal(x, y, err_msg=""):
    try:
        fail_if_array_equal(x, y, err_msg)
    except AssertionError as e:
        details = str(e).strip()
        if err_msg and err_msg not in details:
            details = f"{err_msg}\n{details}"
        raise AssertionError(details) from None

def safe_assert_array_almost_equal(x, y, decimal=6, err_msg=""):
    try:
        assert_array_almost_equal(x, y, decimal, err_msg)
    except AssertionError as e:
        details = str(e).strip()
        if err_msg:
            details = f"{err_msg}\n{details}"
        raise AssertionError(details) from None