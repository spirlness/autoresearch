from unittest.mock import MagicMock, patch

from autoresearch_trainer.train_session import _cleanup_training_attempt


def test_cleanup_training_attempt_closes_resources_and_restores_runtime_state():
    trainer = MagicMock()
    train_loader = MagicMock()

    with patch(
        "autoresearch_trainer.train_session.gc.isenabled", return_value=False
    ), patch("autoresearch_trainer.train_session.gc.enable") as mock_enable, patch(
        "autoresearch_trainer.train_session.gc.collect"
    ) as mock_collect, patch(
        "autoresearch_trainer.train_session.gc.unfreeze", create=True
    ) as mock_unfreeze, patch(
        "autoresearch_trainer.train_session.torch.cuda.is_available",
        return_value=True,
    ), patch(
        "autoresearch_trainer.train_session.torch.cuda.empty_cache"
    ) as mock_empty_cache:
        _cleanup_training_attempt(trainer, train_loader)

    train_loader.close.assert_called_once()
    trainer.close.assert_called_once()
    mock_unfreeze.assert_called_once()
    mock_enable.assert_called_once()
    mock_collect.assert_called_once()
    mock_empty_cache.assert_called_once()


def test_cleanup_training_attempt_tolerates_missing_close_methods():
    with patch(
        "autoresearch_trainer.train_session.gc.isenabled", return_value=True
    ), patch("autoresearch_trainer.train_session.gc.enable") as mock_enable, patch(
        "autoresearch_trainer.train_session.gc.collect"
    ) as mock_collect, patch(
        "autoresearch_trainer.train_session.torch.cuda.is_available",
        return_value=False,
    ):
        _cleanup_training_attempt(object(), None)

    mock_enable.assert_not_called()
    mock_collect.assert_called_once()
