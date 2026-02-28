import unittest
from unittest.mock import MagicMock, patch
from Scripts.trade import execute_trade
from datetime import datetime

class TestSentimentGate(unittest.TestCase):

    @patch('Scripts.trade.ib')
    @patch('Scripts.trade.ensure_ib_connected')
    @patch('Scripts.news.get_ticker_sentiment')
    @patch('Scripts.trade.exit_manager')
    @patch('Scripts.trade.submit_ml_sized_order')
    @patch('Scripts.trade._log_entry_to_db')
    def test_sentiment_suppression(self, mock_log, mock_submit, mock_em, mock_sent, mock_ensure, mock_ib):
        # 1. Negative sentiment should suppress
        mock_sent.return_value = -0.5
        mock_em.is_circuit_breaker_active.return_value = False
        
        pred_results = [('AAPL', 0.8)] # Strong buy
        execute_trade(pred_results, MagicMock())
        
        mock_submit.assert_not_called()
        print("PASS: Negative sentiment suppressed entry")

    @patch('Scripts.trade.ib')
    @patch('Scripts.trade.ensure_ib_connected')
    @patch('Scripts.news.get_ticker_sentiment')
    @patch('Scripts.trade.exit_manager')
    @patch('Scripts.trade.submit_ml_sized_order')
    @patch('Scripts.trade._log_entry_to_db')
    @patch('Scripts.trade.Stock')
    def test_sentiment_allowed(self, mock_stock, mock_log, mock_submit, mock_em, mock_sent, mock_ensure, mock_ib):
        # 2. Positive sentiment should allow
        mock_sent.return_value = 0.5
        mock_em.is_circuit_breaker_active.return_value = False
        mock_em.positions.get.return_value = MagicMock(stop_price=95, target_price=110)
        mock_submit.return_value = 10
        
        # Mock ticker for price
        mock_ticker = MagicMock()
        mock_ticker.last = 100.0
        mock_ib.reqMktData.return_value = mock_ticker
        
        pred_results = [('AAPL', 0.8)] # Strong buy
        execute_trade(pred_results, MagicMock())
        
        mock_submit.assert_called_once()
        # Verify sentiment_score passed to log
        args, kwargs = mock_log.call_args
        self.assertEqual(kwargs['sentiment_score'], 0.5)
        print("PASS: Positive sentiment allowed entry and logged score")

if __name__ == '__main__':
    unittest.main()
