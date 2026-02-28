from dotenv import load_dotenv
load_dotenv()

import unittest
from unittest.mock import MagicMock, patch
from Scripts.trade import submit_ml_sized_order

class TestPositionSizing(unittest.TestCase):

    @patch('Scripts.trade.ib')
    @patch('Scripts.trade.ensure_ib_connected')
    @patch('Scripts.trade.ml_position_sizer')
    @patch('Scripts.trade.Stock')
    @patch('Scripts.trade.MarketOrder')
    @patch('Scripts.utils.get_current_vix')
    def test_multipliers(self, mock_vix, mock_order, mock_stock, mock_sizer, mock_ensure, mock_ib):
        # Base setup
        mock_sizer.calculate_position_size.return_value = 100
        
        test_cases = [
            # (VIX, Regime, ExpectedQty)
            (20, 'bullish', 100),       # 1.0 * 1.0 = 1.0
            (20, 'sideways', 50),      # 1.0 * 0.5 = 0.5
            (20, 'bearish', 25),       # 1.0 * 0.25 = 0.25
            (20, 'volatile', 25),      # 1.0 * 0.25 = 0.25
            (35, 'bullish', 50),       # 0.5 * 1.0 = 0.5
            (35, 'sideways', 25),      # 0.5 * 0.5 = 0.25
            (35, 'bearish', 12),       # 0.5 * 0.25 = 0.125 -> 100 * 0.125 = 12.5 -> 12
            (35, 'unknown', 25),       # 0.5 * 0.5 = 0.25
            (20, 'unknown', 50),       # 1.0 * 0.5 = 0.5
        ]
        
        for vix, regime, expected in test_cases:
            mock_vix.return_value = vix
            qty = submit_ml_sized_order('AAPL', 'buy', 0.7, current_price=150, market_regime=regime)
            self.assertEqual(qty, expected, f"Failed for VIX={vix}, Regime={regime}. Got {qty}, expected {expected}")

    @patch('Scripts.trade.ib')
    @patch('Scripts.trade.ensure_ib_connected')
    @patch('Scripts.trade.ml_position_sizer')
    @patch('Scripts.utils.get_current_vix')
    def test_floor_at_one(self, mock_vix, mock_sizer, mock_ensure, mock_ib):
        # If base qty is small, ensure we still trade at least 1 share if multipliers are small
        mock_sizer.calculate_position_size.return_value = 2
        mock_vix.return_value = 40  # 0.5 mult
        # Regime bearish = 0.25 mult
        # Final mult = 0.125
        # 2 * 0.125 = 0.25 -> Floor at 1
        qty = submit_ml_sized_order('AAPL', 'buy', 0.7, market_regime='bearish')
        self.assertEqual(qty, 1)

if __name__ == '__main__':
    unittest.main()
