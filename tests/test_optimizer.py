from qpe.portfolio_optimizer import PortfolioOptimizer


def test_optimize_basic():
    opt = PortfolioOptimizer(peso_min=0.02, peso_max=0.50)
    scores = [90, 80, 70, 60, 50]
    df = opt.optimize(scores, ["A", "B", "C", "D", "E"])
    assert abs(df["weight_pct"].sum() - 100) < 1
    assert all(df["weight_pct"] >= 1.9)
    assert all(df["weight_pct"] <= 50.1)


def test_optimize_single():
    opt = PortfolioOptimizer()
    df = opt.optimize([100], ["A"])
    assert abs(df["weight_pct"].iloc[0] - 100) < 1


def test_optimize_equal():
    opt = PortfolioOptimizer(peso_min=0.01, peso_max=0.50)
    scores = [80, 80]
    df = opt.optimize(scores, ["A", "B"])
    assert abs(df["weight_pct"].iloc[0] - df["weight_pct"].iloc[1]) < 1


def test_optimize_sorting():
    opt = PortfolioOptimizer(peso_min=0.01, peso_max=0.50)
    scores = [10, 90, 50]
    df = opt.optimize(scores, ["A", "B", "C"])
    assert df.iloc[0]["ticker"] == "B"
    assert df.iloc[-1]["ticker"] == "A"
    assert df.iloc[-1]["ticker"] == "A"


def test_allocate_by_profile():
    opt = PortfolioOptimizer(peso_min=0.05, peso_max=0.40)
    tickers = ["A", "B", "C", "D"]
    scores = [90, 80, 70, 60]
    result = opt.allocate_by_profile(
        tickers, scores,
        {"Ações": 60, "FIIs": 40},
        {"Ações": ["A", "B"], "FIIs": ["C", "D"]},
    )
    assert abs(result["weight_sum"] - 100) < 1
    assert result["total_assets"] == 4
