import json
import csv
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare import clean_comment, email_body, prepare, read_texts, validate_manifest


class PreparationTests(unittest.TestCase):
    def test_chromium_pipeline_truncates_long_comments_without_appending_eos(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "train": 2,
                                             "validation": 3, "participant": 4, "null": 5}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer.save_pretrained(root / "tokenizer")
            (root / "config").mkdir()
            (root / "splits").mkdir()
            (root / "config/experiment.json").write_text(json.dumps(dict(
                domains=dict(chromium=dict(train_documents=1, train_tokens=256, validation_documents=1)))))
            np.savez(root / "splits/chromium_audit.npz", offsets=[0, 1, 2], labels=[1, 0])
            rows = [dict(file="corpus.zip", row=i, role=role, source=source, query_position=position, prediction_tokens=count)
                    for i, (role, source, position, count) in enumerate([(0, 0, -1, 256), (1, 0, -1, 65),
                                                                      (2, 0, 0, 256), (2, 1, 1, 69)])]
            with gzip.open(root / "splits/chromium_documents.csv.gz", "wt") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with zipfile.ZipFile(root / "corpus.zip", "w") as archive:
                texts = ["train " * 300, "validation " * 66, "participant " * 400, "null " * 70]
                archive.writestr("chromium-corpus/utterances.jsonl", "\n".join(json.dumps(dict(text=text)) for text in texts))
            with patch("prepare.ROOT", root):
                prepare("chromium", root, root / "prepared", tokenizer_path=root / "tokenizer")
            with np.load(root / "prepared/chromium.npz") as result:
                np.testing.assert_array_equal(np.diff(result["offsets"]), [257, 66, 257, 70])
                np.testing.assert_array_equal(result["tokens"], np.repeat([2, 3, 4, 5], [257, 66, 257, 70]))

    def test_audit_labels_and_bags_match_training_and_query_assignments(self):
        rows = [dict(role=0, source=1, query_position=-1),
                dict(role=1, source=1, query_position=-1),
                dict(role=2, source=1, query_position=1),
                dict(role=2, source=0, query_position=0)]
        audit = dict(offsets=np.array([0, 1, 2]), labels=np.array([0, 1]))
        validate_manifest(rows, audit)
        with self.assertRaisesRegex(ValueError, "participation labels"):
            validate_manifest(rows, dict(audit, labels=np.array([1, 0])))
        rows[-1]["source"] = 1
        with self.assertRaisesRegex(ValueError, "declared source bags"):
            validate_manifest(rows, audit)

    def test_email_body_preserves_whitespace_and_normalizes_newlines(self):
        raw = b'From: sender@example.org\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n a  b\r\nnext\rline\r\n'
        self.assertEqual(email_body(raw), ' a  b\nnext\nline\n')

    def test_email_skips_multipart_attachments(self):
        raw = (b'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n'
               b'--x\nContent-Type: text/plain\n\nbody\n'
               b'--x\nContent-Type: text/plain\nContent-Disposition: attachment\n\nattached\n--x--\n')
        self.assertEqual(email_body(raw), 'body')

    def test_chromium_removes_quote_lines_only(self):
        self.assertEqual(clean_comment('  > quote\nactual > text\n\t> another\n tail  '), 'actual > text\n tail')

    def test_parquet_locations_preserve_manifest_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pq.write_table(pa.table({'text': ['zero', 'one', 'two', 'three']}), path / 'data.parquet', row_group_size=2)
            rows = [{'file': 'data.parquet', 'row': 3}, {'file': 'data.parquet', 'row': 0}]
            self.assertEqual(dict(read_texts('cc_news', rows, path)), {0: 'three', 1: 'zero'})

    def test_zip_locations_apply_comment_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with zipfile.ZipFile(path / 'corpus.zip', 'w') as archive:
                archive.writestr('chromium-corpus/utterances.jsonl', '\n'.join(json.dumps({'text': text}) for text in ['skip', '> quoted\nkeep']))
            rows = [{'file': 'corpus.zip', 'row': 1}]
            self.assertEqual(list(read_texts('chromium', rows, path)), [(0, 'keep')])


if __name__ == '__main__':
    unittest.main()
