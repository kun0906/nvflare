import os
import torch


class SnapshotManager:
    def __init__(self, snapshot_dir, logger=None):
        self.snapshot_dir = snapshot_dir
        self.logger = logger
        os.makedirs(snapshot_dir, exist_ok=True)

    def save_snapshot(self, model, round_num):
        """Save the model and round number as a snapshot."""
        snapshot_path = os.path.join(self.snapshot_dir, f"snapshot_round_{round_num}.pt")
        torch.save({'model_state_dict': model.state_dict(), 'round_num': round_num}, snapshot_path)
        if self.logger:
            self.logger.info(f"Snapshot saved: {snapshot_path}")
        else:
            print(f"Snapshot saved: {snapshot_path}")

    def load_snapshot(self):
        """Load the latest snapshot."""
        snapshot_files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.pt')]
        if not snapshot_files:
            return None, 0  # No snapshot found

        latest_snapshot = max(snapshot_files, key=lambda f: int(f.split('_')[2]))
        snapshot_path = os.path.join(self.snapshot_dir, latest_snapshot)
        snapshot = torch.load(snapshot_path)
        if self.logger:
            self.logger.info(f"Snapshot loaded: {snapshot_path}")
        else:
            print(f"Snapshot loaded: {snapshot_path}")

        return snapshot['model_state_dict'], snapshot['round_num']

    def has_snapshot(self):
        """Check if there are any snapshots saved."""
        snapshot_files = [f for f in os.listdir(self.snapshot_dir) if f.endswith('.pt')]
        return len(snapshot_files) > 0

