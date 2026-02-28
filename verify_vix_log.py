import unittest
from unittest.mock import MagicMock, patch
from Scripts.trade import execute_trade

class TestVixLogging(unittest.TestCase):

    @patch('Scripts.trade.ib')
    @patch('Scripts.trade.ensure_ib_connected')
    @patch('Scripts.news.get_ticker_sentiment')
    @patch('Scripts.utils.get_current_vix')
    @patch('Scripts.trade.exit_manager')
    @patch('Scripts.trade.submit_ml_sized_order')
    @patch('Scripts.trade._log_entry_to_db')
    @patch('Scripts.trade.Stock')
    def test_vix_passed_to_log(self, mock_stock, mock_log, mock_submit, mock_em, mock_vix, mock_sent, mock_ensure, mock_ib):
        # Setup
        mock_vix.return_value = 25.0
        mock_sent.return_value = 0.1
        mock_em.is_circuit_breaker_active.return_value = False
        mock_em.positions.get.return_value = MagicMock()
        mock_submit.return_value = 10
        
        mock_ticker = MagicMock()
        mock_ticker.last = 100.0
        mock_ib.reqMktData.return_value = mock_ticker
        
        # Execute
        execute_trade([('AAPL', 0.8)], MagicMock())
        
        # Verify
        args, kwargs = mock_log.call_args
        self.assertEqual(kwargs['vix_score'], 25.0)
        print("PASS: VIX score correctly passed to _log_entry_to_db")

if __name__ == '__main__':
    unittest.main()
