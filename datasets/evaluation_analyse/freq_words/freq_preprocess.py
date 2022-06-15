import argparse
import os
import json
from collections import Counter
from copy import copy

import jieba
import stopwordsiso as stopwords
import re, string

LANGS = "af am ar az be bg bn br bs ca cs cy da de el en es et fa fi fr fy ga gd gl gu ha he hi hr hu hy id is it ja jv ka kk km kn ko lo lt lv mg mk ml mn mr ms my ne nl no or pa pl ps pt ro ru sd si sk sl so sq sr su sv sw ta th tl tr uk ur uz vi xh yi zh"
LANGS = LANGS.split()  # type: List[str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default=LANGS,
                        help="Languages loaded from the annotations path (lg1 lg2 lg3 .. ex: en fr es de). By default, use all languages.")
    parser.add_argument("--task", type=str,
                        help="The task name in zero-shot annotations.")
    parser.add_argument("--mode", type=str,
                        help="The mode name for pretraining: random or vtlm.")
    parser.add_argument("--translation_mode", type=str, default="filter",
                        help="The mode name for pretraining: filter or full.")
    parser.add_argument("--eval_mode", type=str,
                        help="The eval mode: zero_shot or multilingual.")
    parser.add_argument("--pretrain_model", type=str,
                        help="The pretrain model: TTMML or XUNITER.")
    parser.add_argument("--task_eval_test", type=str,
                        help="The task name in zero-shot test results.")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="pretrain batch size")
    parser.add_argument("--steps", default=None, type=int,
                        help="pretrain steps for the specific step")
    parser.add_argument("--pretrain_langs", default=20, type=int,
                        help="total languages in pretrain_langs")
    parser.add_argument("--annotation_path", type=str,
                        help="The translation file from the annotations path.")
    parser.add_argument("--eval_annotation_path", type=str,
                        help="The evaluation annotation files.")
    parser.add_argument("--eval_en_path", type=str,
                        help="The evaluation annotation files.")
    parser.add_argument("--eval_test_path", type=str,
                        help="The evaluation-test files.")
    parser.add_argument("--freq_output_path", type=str,
                        help="The frequence file for the histogram of the tokens based on the training dataset.")
    parser.add_argument("--score_output_path", type=str,
                        help="The score file from frequence and evaluation results.")
    return parser.parse_args()


def remove_punctuation(sentence):
    if not isinstance(sentence, str):
        print("sentence {} is not string!".format(sentence))
        return None

    punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
    exclude = set(string.punctuation)
    out = ''.join(ch for ch in sentence if ch not in exclude)
    return re.sub(r"[%s]+" % punc, "", out).strip()


class PreprocessMultilingualBatch(object):
    def __init__(
            self,
            args,
            langs,
            split,
            pretrain_batch_size,
            pretrain_steps,
            mode
    ):
        self.langs = langs
        self.split = split
        self.task = args.task
        self.mode = args.mode
        self.eval_mode = args.eval_mode
        self.pretrain_model = args.pretrain_model
        self.task_eval_test = args.task_eval_test
        self.annotation_path = args.annotation_path
        self.eval_annotation_path = args.eval_annotation_path
        self.eval_en_path = args.eval_en_path
        self.eval_test_path = args.eval_test_path
        self.freq_output_path = args.freq_output_path
        self.score_output_path = args.score_output_path
        self.captions = self.load_captions(path=self.annotation_path, split=self.split, langs=self.langs)
        self.all_stop_words = self.load_stopwords()
        self.translation_mode = args.translation_mode
        self.eval_captions = self.load_eval_captions(self.eval_annotation_path, self.eval_en_path, task=self.task,
                                                     langs=self.langs)
        self.eval_results = self.load_eval_results(self.eval_test_path, self.task_eval_test, self.langs,
                                                   pretrain_batch_size, pretrain_steps, mode, 20)

        freq_file = os.path.join(self.freq_output_path, f"cc_pretrain_freq_words_{self.task}.json")
        scores_file = os.path.join(self.score_output_path,
                                   f"{self.pretrain_model}_eval_test_scores_{self.task}_{self.eval_mode}_{self.mode}.json")
        if os.path.exists(freq_file) and os.path.exists(scores_file):
            self.freq_dict = self.freq_read(freq_file)
            self.scores_dict = self.freq_read(scores_file)

    def get_freq(self):
        freq_ = dict()

        for lang_ in self.langs:

            translated_sents = [
                remove_punctuation(sentence)
                for lang, captions_lang in self.captions.items()
                for id, sentence in captions_lang.items()
                if lang in lang_
            ]
            if lang_ in ['zh']:
                translated_sents = [' '.join(list(jieba.cut(m))) for m in translated_sents]
            print("translated_sents demo: ",translated_sents[:10])
            filter_translated_sents = [self.del_stopwords(m.split(), lang_) for m in translated_sents]
            flatten_sents = [i for item in filter_translated_sents for i in item]
            counter_sents = Counter(flatten_sents)
            freq_[lang_] = counter_sents

        with open(os.path.join(self.freq_output_path,
                               f"cc_pretrain_freq_words_{self.task}.json"),"w") as f:
            f.write(json.dumps(freq_, ensure_ascii=False, indent=4, separators=(',', ':')))
        return freq_

    def get_scores(self):

        freq_file = os.path.join(self.freq_output_path, f"cc_pretrain_freq_words_{self.task}.json")
        if not os.path.exists(freq_file):
            self.freqs_ = self.get_freq()
        else:
            self.freqs_ = self.freq_read(freq_file)

        scores_all = dict()
        for lang_ in self.langs:
            print(list(self.freqs_[lang_].items())[:30])
            id2sents_per_lang = dict()
            sents2id_per_lang = dict()
            score_per_lang = dict()

            if lang_ not in ['en']:
                id_sentences_pairs = [
                    [multiple_elements['id'], multiple_elements['caption']]
                    for lang, captions_lang in self.eval_captions.items()
                    for item, multiple_elements in captions_lang.items()
                    if lang in lang_
                ]
            else:
                id_sentences_pairs = [
                    [multiple_elements['identifier'], multiple_elements['sentence']]
                    for lang, captions_lang in self.eval_captions.items()
                    for item, multiple_elements in captions_lang.items()
                    if lang in lang_
                ]
            print(id_sentences_pairs[:10])
            for item in id_sentences_pairs:
                id2sents_per_lang[item[0]] = item[1]
                sents2id_per_lang[item[1]] = item[0]

            for idx, item in enumerate(self.eval_results[lang_]):
                result = 0 if item['label'] == item['prediction'] else 1
                captions = remove_punctuation(item['sentence'])
                if lang_ in ['zh']:
                    captions = ' '.join(list(jieba.cut(captions)))
                captions_filter = self.del_stopwords([m for m in captions.split()], lang_)
                example_id = sents2id_per_lang[item['sentence']]
                avg_ = [self.freqs_[lang_][m] if m in self.freqs_[lang_].keys() else 0 for m in captions_filter]
                avg_mins_ = copy(avg_)
                avg_mins_.sort()
                avg_min_l = avg_mins_[:2] if len(avg_mins_) >= 2 else [0, 0]
                score_per_lang[example_id] = [{'label': item['label'], 'prediction': item['prediction'],
                                               'result': result, 'score_avg': str(sum(avg_) / len(avg_)),
                                               'score_avg_mins': str(sum(avg_min_l) / 2),
                                               'frequence': ' '.join([str(m) for m in avg_]),
                                               'caption': ' '.join(captions_filter), 'sentences': item['sentence']}]
            scores_all[lang_] = score_per_lang
        if self.eval_mode == 'translate_train':
            with open(os.path.join(self.score_output_path,
                                   f"{self.pretrain_model}_eval_test_scores_{self.task}_{self.eval_mode}_{self.mode}_{self.translation_mode}.json"),
                      "w") as f:
                f.write(json.dumps(scores_all, ensure_ascii=False, indent=4, separators=(',', ':')))
        else:
            with open(os.path.join(self.score_output_path,
                                   f"{self.pretrain_model}_eval_test_scores_{self.task}_{self.eval_mode}_{self.mode}.json"),
                      "w") as f:
                f.write(json.dumps(scores_all, ensure_ascii=False, indent=4, separators=(',', ':')))

    def del_stopwords(self, words_, lang_):
        return [m for m in words_ if m not in self.all_stop_words[lang_]]

    @staticmethod
    def load_captions(path, split, langs):
        def load1(lang):
            with open(os.path.join(path, f"{lang}-{split}.json")) as f:
                return json.load(f)

        return {lang: load1(lang) for lang in langs}

    def load_stopwords(self):
        # https://github.com/stopwords-iso/stopwords-iso
        all_stop_words = dict()
        for lang_ in self.langs:
            all_stop_words[lang_] = list(stopwords.stopwords(lang_))
        return all_stop_words

    @staticmethod
    def load_eval_captions(path, en_path, task, langs):
        def load1(lang):
            item_dict = {}
            if lang != 'en':
                data = [json.loads(line) for line in open(os.path.join(path, f"{task}-{lang}.jsonl"), 'r')]
                for ln in data:
                    item_dict[ln['id']] = ln
            else:
                data = [json.loads(line) for line in open(os.path.join(en_path, "test.jsonl"), 'r')]
                for ln in data:
                    item_dict[ln['identifier']] = ln
            return item_dict

        return {lang: load1(lang) for lang in langs}

    def load_eval_results(self, path, task, langs, pretrain_batch_size, pretrain_steps, mode, total_langs):
        def load1(lang):
            if self.eval_mode == 'translate_train':
                with open(os.path.join(path,
                                       f"{task}{lang}-{mode}-batch_size_{pretrain_batch_size}-step_{pretrain_steps}-langs_{total_langs}-eval_step_24291-{self.translation_mode}",
                                       f"pytorch_model_epoch_2_step_24291.bin-", "test_result.json")) as f:
                    return json.load(f)
            else:
                with open(os.path.join(path,
                                       f"{task}{lang}-{mode}-batch_size_{pretrain_batch_size}-step_{pretrain_steps}-langs_{total_langs}",
                                       f"pytorch_model_best.bin-", "test_result.json")) as f:
                    return json.load(f)

        return {lang: load1(lang) for lang in langs}

    @staticmethod
    def freq_read(freq_file):
        with open(freq_file, "r") as f:
            return json.load(f)


'''
eval_en_file:
    {"validation": {"28": "False"}, "sentence": "There is an empty glass.", "left_url": "http://www.belgiansmaak.com/wp-content/gallery/belgian-beers-post/dynamic/31.-Tilquin.jpg-nggid041318-ngg0dyn-0x0x100-00f0w010c010r110f110r010t010.jpg", "writer": "103", "label": "False", "right_url": "https://www.craftbrewingbusiness.com/wp-content/uploads/2017/09/unnamed-1.jpg", "synset": "beer bottle", "query": "group of beer bottles41", "identifier": "test1-0-1-0", "extra_validations": {"56": "False", "83": "False", "19": "False", "92": "False"}}
eval_id_file:
    {"concept": "39-Panci", "language": "id", "caption": "Panci di salah satu foto berada di atas kompor yang tidak menyala, sedangkan di foto lainnya, api di bawah panci menyala.", "left_img": "39-0.jpg", "right_img": "39-11.jpg", "annotator_info": {"annotator_id": "id_01", "country_of_birth": "Indonesia", "country_of_residence": "Indonesia", "gender": "male", "age": 31}, "chapter": "Basic actions and technology", "id": "id-0", "label": true}
# eval_test_results_file:
    [{"sentence": "Panci di salah satu foto berada di atas kompor yang tidak menyala, sedangkan di foto lainnya, api di bawah panci menyala.", "prediction": 0, "label": 1}]
'''


def main():
    args = parse_args()

    if not isinstance(args.langs, list):
        args.langs = args.langs.split('-')
    print("pretrain langs".format(args.langs))

    process = PreprocessMultilingualBatch(args, args.langs, "train", args.batch_size, args.steps, args.mode)
    process.get_scores()


if __name__ == "__main__":
    main()
