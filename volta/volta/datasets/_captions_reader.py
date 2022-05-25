
import os
import json
class MultilingualCaptionReader(object):
    def __init__(self, caption_path, split, langs):
        self.caption_path = caption_path
        self.split = split
        self.langs = langs
        self.captions = self.load_captions()

    def load_captions(self):
        def load1(lang):
            with open(os.path.join(self.caption_path, f"{lang}-{self.split}.json")) as f:
                return json.load(f)
        return {lang: load1(lang) for lang in self.langs}