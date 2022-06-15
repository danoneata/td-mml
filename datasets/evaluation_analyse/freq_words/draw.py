import argparse
import pdb
import random
import os
import json

from matplotlib import pyplot as plt
from matplotlib.pyplot import savefig
import streamlit as st
import numpy as np

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
    parser.add_argument("--translation_mode", type=str, default=None,
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
    parser.add_argument("--freq_path", type=str,
                        help="The frequence file for the histogram of the tokens based on the training dataset.")
    parser.add_argument("--score_path", type=str,
                        help="The score file from frequence and evaluation results.")
    parser.add_argument("--figs_path", type=str,
                        help="The score file from frequence and evaluation results.")
    return parser.parse_args()

class DrawLoader(object):
    def __init__(
            self,
            args,
            langs
    ):
        self.langs = langs
        self.task = args.task
        self.mode = args.mode
        self.eval_mode = args.eval_mode
        self.pretrain_model = args.pretrain_model
        self.task_eval_test = args.task_eval_test
        self.freq_path = args.freq_path
        self.score_path = args.score_path
        self.figs_path = args.figs_path
        self.translation_mode = args.translation_mode
        freq_file = os.path.join(self.freq_path, f"cc_pretrain_freq_words_{self.task}.json")
        if self.translation_mode is None:
            scores_file = os.path.join(self.score_path,
                                   f"{self.pretrain_model}_eval_test_scores_{self.task}_{self.eval_mode}_{self.mode}.json")
        else:
            scores_file = os.path.join(self.score_path,
                                   f"{self.pretrain_model}_eval_test_scores_{self.task}_{self.eval_mode}_{self.mode}_{self.translation_mode}.json")
        if os.path.exists(freq_file) and os.path.exists(scores_file):
            self.freq_dict = self.freq_read(freq_file)
            self.scores_dict = self.freq_read(scores_file)

    def get_frac_correct(self,i,idxs,results):
        r = results[idxs == i]
        return 100 * r.sum() / len(r)

    def draw_per_lang(self,marvl_results_scores, lang_):
        results = np.array([m[0] for m in marvl_results_scores])
        scores = np.array([float(m[1]) for m in marvl_results_scores])

        num_bins = 10
        _, bins = np.histogram(scores, bins=num_bins)
        idxs = np.digitize(scores, bins)

        frac_correct = [self.get_frac_correct(i,idxs,results) for i in range(1, num_bins + 1)]
        fig, ax = plt.subplots()
        width = 0.8 * (bins[1] - bins[0])
        ax.bar(bins[1:], frac_correct, align="edge", width=width)
        ax.set_xlabel("similarity to train data")
        ax.set_ylabel("fraction correct (%)")
        st.pyplot(fig)

        if self.translation_mode is None:
            savefig(os.path.join(self.figs_path,"{}_{}_{}".format(self.pretrain_model,self.task,self.eval_mode),"{}_{}_{}_{}_{}.jpg".format(self.pretrain_model, self.task, self.mode, self.eval_mode, lang_)))
        else:
            savefig(os.path.join(self.figs_path,"{}_{}_{}".format(self.pretrain_model,self.task,self.eval_mode), "{}_{}_{}_{}_{}_{}.jpg".format(self.pretrain_model,
                    self.task, self.mode, self.eval_mode, lang_, self.translation_mode)))

    def draw(self):
        for lang_ in self.langs:
            marvl_results_scores = [
                ((value['result']), value['score_avg_mins'])
                for id, value in enumerate(self.scores_dict[lang_])
            ]
            self.draw_per_lang(marvl_results_scores, lang_)

    @staticmethod
    def freq_read(freq_file):
        with open(freq_file, "r", encoding='utf-8') as f:
            return json.load(f)

def main():
    args = parse_args()

    if not isinstance(args.langs, list):
        args.langs = args.langs.split('-')
    print("pretrain langs".format(args.langs))

    draw_data = DrawLoader(args, args.langs)
    draw_data.draw()

if __name__ == "__main__":
    main()