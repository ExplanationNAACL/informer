from flask import Flask, request
from flask import render_template
import random
import sys

sys.path.append("../..")
from pipeline import Informer
from salience_basic_util import SEQ_TASK, NLI_TASK, CORPUS_LEVEL, INSTANCE_LEVEL

# LIME demo imports
from lit_nlp.examples.datasets import glue
from lit_nlp.examples.models import glue_models
from informers_specs_util import SST2LIME, MNLILIME
from lit_nlp.components import lime_explainer

app = Flask(__name__)
informer_obj = None
perturbations = [
    'The movie constitute very fascinate and rum and I urge for everyone to bribe tickets and visualise it',
    'The movie personify real witching and risible and I recommend for everyone to steal tickets and ascertain it',
    'The movie personify really enamor and risible and I commend for everyone to bargain tickets and see it',
    'The movie comprise really enamor and risible and I recommend for everyone to buy tickets and see it',
    'The movie constitute very enamor and risible and I commend for everyone to bribe tickets and ascertain it',
    'The movie is rattling charming and funny and I urge for everyone to buy tickets and visualise it',
    'The movie embody real trance and rummy and I urge for everyone to bargain tickets and envision it',
    'The movie personify identical fascinate and rummy and I urge for everyone to steal tickets and ascertain it',
    'The movie personify really enamor and risible and I commend for everyone to bribe tickets and visualise it',
    'The movie constitute real witching and risible and I recommend for everyone to corrupt tickets and ascertain it',
    'The movie comprise real fascinate and risible and I urge for everyone to buy tickets and visualise it',
    'The movie constitute real charming and singular and I recommend for everyone to corrupt tickets and visualise it',
    'The movie constitute real enamor and rummy and I urge for everyone to corrupt tickets and see it',
    'The movie personify very enamor and rummy and I commend for everyone to corrupt tickets and envision it',
    'The movie comprise very trance and rummy and I advocate for everyone to bribe tickets and envision it',
    'The movie is really enamor and singular and I recommend for everyone to bargain tickets and see it',
    'The movie is rattling fascinate and risible and I commend for everyone to bribe tickets and ascertain it',
    'The movie comprise very enamor and singular and I urge for everyone to buy tickets and see it',
    'The movie is very witching and rummy and I recommend for everyone to buy tickets and ascertain it',
    'The movie personify very fascinate and rummy and I advocate for everyone to corrupt tickets and envision it',
    'The movie embody real trance and singular and I recommend for everyone to corrupt tickets and see it',
    'The movie personify real charming and rum and I commend for everyone to corrupt tickets and ascertain it',
    'The movie is very trance and rummy and I commend for everyone to buy tickets and visualise it',
    'The movie constitute really enamor and rum and I commend for everyone to steal tickets and envision it',
    'The movie personify rattling witching and rummy and I recommend for everyone to bribe tickets and ascertain it',
    'The movie comprise identical charming and risible and I recommend for everyone to bargain tickets and see it',
    'The movie comprise very trance and rum and I recommend for everyone to buy tickets and visualise it',
    'The movie comprise identical fascinate and risible and I commend for everyone to bribe tickets and ascertain it',
    'The movie constitute real witching and rummy and I recommend for everyone to corrupt tickets and visualise it',
    'The movie personify really trance and funny and I recommend for everyone to corrupt tickets and ascertain it',
    'The movie personify very fascinate and singular and I recommend for everyone to steal tickets and visualise it',
    'The movie embody really charming and singular and I urge for everyone to corrupt tickets and see it',
    'The movie embody really witching and singular and I urge for everyone to corrupt tickets and visualise it',
    'The movie is identical fascinate and risible and I urge for everyone to bargain tickets and envision it',
    'The movie embody real trance and rummy and I commend for everyone to corrupt tickets and ascertain it',
    'The movie is rattling witching and risible and I urge for everyone to steal tickets and ascertain it',
    'The movie personify rattling witching and risible and I urge for everyone to bribe tickets and see it',
    'The movie comprise rattling enamor and singular and I advocate for everyone to buy tickets and ascertain it',
    'The movie personify identical enamor and funny and I recommend for everyone to buy tickets and envision it',
    'The movie constitute rattling enamor and singular and I urge for everyone to corrupt tickets and envision it',
    'The movie embody rattling charming and funny and I commend for everyone to steal tickets and visualise it',
    'The movie comprise rattling witching and singular and I urge for everyone to bargain tickets and ascertain it',
    'The movie personify real fascinate and risible and I advocate for everyone to bribe tickets and envision it',
    'The movie constitute very enamor and funny and I urge for everyone to bargain tickets and envision it',
    'The movie is real enamor and rummy and I advocate for everyone to bribe tickets and ascertain it',
    'The movie constitute identical enamor and risible and I urge for everyone to buy tickets and ascertain it',
    'The movie comprise real enamor and rum and I advocate for everyone to buy tickets and visualise it',
    'The movie embody really enamor and rum and I recommend for everyone to bribe tickets and envision it',
    'The movie embody real trance and funny and I recommend for everyone to buy tickets and visualise it',
    'The movie embody really trance and rum and I commend for everyone to bribe tickets and ascertain it',
    'The movie personify rattling enamor and funny and I urge for everyone to corrupt tickets and see it',
    'The movie comprise real fascinate and funny and I urge for everyone to bargain tickets and visualise it',
    'The movie embody really charming and funny and I commend for everyone to bargain tickets and ascertain it',
    'The movie constitute very trance and rummy and I urge for everyone to corrupt tickets and ascertain it',
    'The movie embody really enamor and rum and I commend for everyone to bribe tickets and visualise it',
    'The movie personify real fascinate and rum and I urge for everyone to corrupt tickets and ascertain it',
    'The movie embody identical enamor and rummy and I advocate for everyone to bribe tickets and ascertain it',
    'The movie comprise really charming and rummy and I recommend for everyone to corrupt tickets and envision it',
    'The movie is very charming and rum and I recommend for everyone to buy tickets and see it',
    'The movie comprise real witching and rummy and I urge for everyone to bargain tickets and see it',
    'The movie is really enamor and singular and I commend for everyone to bargain tickets and envision it',
    'The movie comprise really charming and risible and I advocate for everyone to steal tickets and envision it',
    'The movie constitute really enamor and rum and I urge for everyone to bribe tickets and visualise it',
    'The movie embody really fascinate and singular and I commend for everyone to buy tickets and ascertain it',
    'The movie embody real charming and rummy and I urge for everyone to steal tickets and visualise it',
    'The movie comprise really enamor and risible and I recommend for everyone to bargain tickets and visualise it',
    'The movie constitute really charming and risible and I recommend for everyone to bargain tickets and ascertain it',
    'The movie embody rattling enamor and risible and I advocate for everyone to buy tickets and visualise it',
    'The movie embody real charming and rummy and I recommend for everyone to buy tickets and ascertain it',
    'The movie embody really witching and funny and I recommend for everyone to bargain tickets and see it',
    'The movie comprise really fascinate and singular and I recommend for everyone to steal tickets and envision it',
    'The movie comprise really witching and funny and I urge for everyone to steal tickets and ascertain it',
    'The movie embody real charming and singular and I recommend for everyone to corrupt tickets and see it',
    'The movie personify identical charming and risible and I commend for everyone to corrupt tickets and see it',
    'The movie comprise identical charming and singular and I commend for everyone to corrupt tickets and ascertain it',
    'The movie embody very trance and rummy and I urge for everyone to bargain tickets and envision it',
    'The movie is identical charming and rum and I commend for everyone to bribe tickets and envision it',
    'The movie comprise really fascinate and rum and I advocate for everyone to corrupt tickets and see it',
    'The movie constitute real fascinate and funny and I advocate for everyone to corrupt tickets and ascertain it',
    'The movie embody very charming and rummy and I urge for everyone to buy tickets and envision it',
    'The movie comprise real trance and funny and I urge for everyone to steal tickets and see it',
    'The movie constitute real fascinate and rummy and I recommend for everyone to bribe tickets and visualise it',
    'The movie is very charming and rum and I recommend for everyone to bribe tickets and envision it',
    'The movie is rattling charming and singular and I advocate for everyone to bribe tickets and see it',
    'The movie embody very trance and singular and I recommend for everyone to steal tickets and ascertain it',
    'The movie personify really charming and risible and I recommend for everyone to bribe tickets and ascertain it',
    'The movie embody real trance and singular and I commend for everyone to corrupt tickets and visualise it',
    'The movie is rattling trance and funny and I recommend for everyone to bribe tickets and visualise it',
    'The movie is really charming and risible and I recommend for everyone to bargain tickets and envision it',
    'The movie constitute very witching and singular and I recommend for everyone to steal tickets and ascertain it',
    'The movie embody real witching and funny and I recommend for everyone to bargain tickets and see it',
    'The movie comprise really witching and singular and I recommend for everyone to bargain tickets and see it',
    'The movie constitute rattling charming and risible and I recommend for everyone to buy tickets and ascertain it',
    'The movie personify identical enamor and rum and I recommend for everyone to steal tickets and ascertain it',
    'The movie comprise very trance and rummy and I urge for everyone to steal tickets and envision it',
    'The movie is very fascinate and rummy and I urge for everyone to buy tickets and ascertain it',
    'The movie comprise real witching and singular and I advocate for everyone to corrupt tickets and see it',
    'The movie personify very fascinate and risible and I advocate for everyone to corrupt tickets and ascertain it',
    'The movie personify identical trance and funny and I advocate for everyone to steal tickets and see it',
    'The movie personify rattling trance and risible and I recommend for everyone to bargain tickets and see it']
sample_data = ["it 's slow -- very , very slow .", "a sometimes tedious film .",
               "or doing last year 's taxes with your ex-wife .",
               "even horror fans will most likely not find what they 're seeking with trouble every day ; the movie lacks both thrills and humor .",
               "in its best moments , resembles a bad high school production of grease , without benefit of song .",
               "the iditarod lasts for days - this just felt like it did .",
               "the action switches between past and present , but the material link is too tenuous to anchor the emotional connections that purport to span a 125-year divide .",
               "a sequence of ridiculous shoot - 'em - up scenes .",
               "( w ) hile long on amiable monkeys and worthy environmentalism , jane goodall 's wild chimpanzees is short on the thrills the oversize medium demands .",
               "it 's just disappointingly superficial -- a movie that has all the elements necessary to be a fascinating , involving character study , but never does more than scratch the surface .",
               "this is a story of two misfits who do n't stand a chance alone , but together they are magnificent .",
               "the script kicks in , and mr. hartley 's distended pace and foot-dragging rhythms follow .",
               "one of creepiest , scariest movies to come along in a long , long time , easily rivaling blair witch or the others .",
               "it 's one pussy-ass world when even killer-thrillers revolve around group therapy sessions .",
               "we know the plot 's a little crazy , but it held my interest from start to finish .",
               "byler reveals his characters in a way that intrigues and even fascinates us , and he never reduces the situation to simple melodrama .",
               "what the director ca n't do is make either of val kilmer 's two personas interesting or worth caring about .",
               "too often , the viewer is n't reacting to humor so much as they are wincing back in repugnance .",
               "it seems to me the film is about the art of ripping people off without ever letting them consciously know you have done so",
               "turns potentially forgettable formula into something strangely diverting .",
               "in the end , we are left with something like two ships passing in the night rather than any insights into gay love , chinese society or the price one pays for being dishonest .",
               "it 's not the ultimate depression-era gangster movie .",
               "the talented and clever robert rodriguez perhaps put a little too much heart into his first film and did n't reserve enough for his second .",
               "it takes talent to make a lifeless movie about the most heinous man who ever lived .",
               "there seems to be no clear path as to where the story 's going , or how long it 's going to take to get there .",
               "the end result is a film that 's neither .", "rarely has leukemia looked so shimmering and benign .",
               "does n't offer much besides glib soullessness , raunchy language and a series of brutal set pieces ... that raise the bar on stylized screen violence .",
               "a celebration of quirkiness , eccentricity , and certain individuals ' tendency to let it all hang out , and damn the consequences .",
               "there ought to be a directing license , so that ed burns can have his revoked .", "bad .",
               "falls neatly into the category of good stupid fun .",
               "to say this was done better in wilder 's some like it hot is like saying the sun rises in the east .",
               "nothing 's at stake , just a twisty double-cross you can smell a mile away -- still , the derivative nine queens is lots of fun .",
               "of course , by more objective measurements it 's still quite bad .",
               "this movie seems to have been written using mad-libs .",
               "i 'd have to say the star and director are the big problems here .",
               "it appears that something has been lost in the translation to the screen .",
               "the film tunes into a grief that could lead a man across centuries .",
               "stealing harvard is evidence that the farrelly bros. -- peter and bobby -- and their brand of screen comedy are wheezing to an end , along with green 's half-hearted movie career .",
               "a full world has been presented onscreen , not some series of carefully structured plot points building to a pat resolution ."]


@app.route("/")
def render_main():
    return render_template("main.html", current_page="Task")


@app.route("/dataset", methods=['POST'])
def dataset():
    data = request.form["input_dropdown"]
    if data == "binary":
        possible_dataset = ["SST-2 validation", "SST-2 train"]
    if data == "ternary":
        possible_dataset = ["Twitter Sentiment Extraction"]
    if data == "nli":
        possible_dataset = ["MNLI train", "MNLI-mismatched validation", "MNLI-matched validation"]

    return render_template("page_two.html", current_page="Dataset", task=data, possible_dataset=possible_dataset)


@app.route("/config", methods=['POST'])
def config():
    data = request.form["input_dropdown"]
    task = request.form["task"]
    if data in ["SST-2 validation", "SST-2 train"]:
        possible_model = ["SST-2 tiny BERT", "SST-2 small BERT"]
    elif data in ["MNLI train", "MNLI-mismatched validation", "MNLI-matched validation"]:
        possible_model = ["MNLI-mBERT"]
    elif data in ["Twitter Sentiment Extraction"]:
        possible_model = ["Twitter-BERT"]

    return render_template("page_three.html", current_page="Model", task=task, possible_dataset=data,
                           possible_model=possible_model)


@app.route("/config_two", methods=['POST'])
def config_two():
    model = request.form["input_dropdown"]
    task = request.form["task"]
    dataset = request.form["dataset"]

    return render_template("page_four.html", current_page="Explanation Method", task=task, dataset=dataset, model=model)


@app.route("/final_page", methods=['POST'])
def final_page():
    model = request.form["model"]
    task = request.form["task"]
    dataset = request.form["dataset"]
    explainer = request.form["input_dropdown"]
    options_dict = {"task": task, "dataset": dataset, "model": model, "explainer": explainer}
    # binary sentiment classification configuration
    config_1 = {"task": 'binary', "dataset": 'SST-2 validation', "model": 'SST-2 tiny BERT', "explainer": 'LIME'}
    config_4 = {"task": 'binary', "dataset": 'SST-2 train', "model": 'SST-2 tiny BERT', "explainer": 'LIME'}
    config_5 = {"task": 'binary', "dataset": 'SST-2 validation', "model": 'SST-2 small BERT', "explainer": 'LIME'}
    config_6 = {"task": 'binary', "dataset": 'SST-2 train', "model": 'SST-2 small BERT', "explainer": 'LIME'}

    # multi-genre nli (mnli) configuration
    config_2 = {"task": 'nli', "dataset": "MNLI train", "model": "MNLI-mBERT", "explainer": 'LIME'}
    config_7 = {"task": 'nli', "dataset": "MNLI-mismatched validation", "model": "MNLI-mBERT", "explainer": 'LIME'}
    config_8 = {"task": 'nli', "dataset": "MNLI-matched validation", "model": "MNLI-mBERT", "explainer": 'LIME'}
    # twitter configuration
    config_3 = {"task": 'ternary', "dataset": "Twitter Sentiment Extraction", "model": "Twitter-BERT",
                "explainer": 'LIME'}
    global informer_obj

    if options_dict == config_1:
        lit_wrapper = SST2LIME(glue_models.SST2Model('../models/sst2-tiny-bert'),
                               glue.SST2Data('validation'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explanation_method,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=SEQ_TASK,
                                num_classes=2,
                                scores_path="../serialized_data/SST-tiny-bert/lime/val.json",
                                preds_path="../serialized_data/SST-tiny-bert/predictions_val.json")
    elif options_dict == config_4:
        lit_wrapper = SST2LIME(glue_models.SST2Model('../models/sst2-tiny-bert'),
                               glue.SST2Data('train'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explanation_method,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=SEQ_TASK,
                                num_classes=2,
                                scores_path="../serialized_data/SST-tiny-bert/lime/train_10k.json",
                                preds_path="../serialized_data/SST-tiny-bert/predictions_train_10k.json")
    elif options_dict == config_5:
        lit_wrapper = SST2LIME(glue_models.SST2Model('../models/sst2-small-bert'),
                               glue.SST2Data('validation'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explanation_method,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=SEQ_TASK,
                                num_classes=2,
                                scores_path="../serialized_data/SST-small-bert/lime/val_unnormalized.json",
                                preds_path="../serialized_data/SST-small-bert/predictions_val_unnormalized.json")
    elif options_dict == config_6:
        lit_wrapper = SST2LIME(glue_models.SST2Model('../models/sst2-small-bert'),
                               glue.SST2Data('train'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explanation_method,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=SEQ_TASK,
                                num_classes=2,
                                scores_path="../serialized_data/SST-small-bert/lime/train_10k_unnormalized.json",
                                preds_path="../serialized_data/SST-small-bert/predictions_train_10k_unnormalized.json")
    elif options_dict == config_2:
        lit_wrapper = MNLILIME(glue_models.MNLIModel('../models/mbert-mnli'),
                               glue.MNLIData('validation_matched'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explainer,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=NLI_TASK,
                                num_classes=3,
                                scores_path="../serialized_data/MNLI-mbert/lime/train_1600.json",
                                preds_path="../serialized_data/MNLI-mbert/predictions_train_1600.json")
    elif options_dict == config_7:
        lit_wrapper = MNLILIME(glue_models.MNLIModel('../models/mbert-mnli'),
                               glue.MNLIData('validation_matched'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explainer,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=NLI_TASK,
                                num_classes=3,
                                scores_path="../serialized_data/MNLI-mbert/lime/val_mismatched_400.json",
                                preds_path="../serialized_data/MNLI-mbert/predictions_val_mismatched_400.json")
    elif options_dict == config_8:
        lit_wrapper = MNLILIME(glue_models.MNLIModel('../models/mbert-mnli'),
                               glue.MNLIData('validation_matched'),
                               lime_explainer.LIME())
        informer_obj = Informer(explainer=lit_wrapper.explainer,
                                dataset=lit_wrapper.get_dataset(),
                                predictor=lit_wrapper.predict_fn,
                                task=NLI_TASK,
                                num_classes=3,
                                scores_path="../serialized_data/MNLI-mbert/lime/val_matched_400.json",
                                preds_path="../serialized_data/MNLI-mbert/predictions_val_matched_400.json")
    elif options_dict == config_3:
        pass
    else:
        print("Strings don't match up somewhere when checking configuration selection")

    return render_template("final_page.html", current_page="Analysis", options=options_dict)


@app.route("/dc", methods=['POST'])
def dc():
    """
    Interface with dataset_consistency class, get all needed output that you would like to put for
    DC metric on the webpage.
    """
    scatterplot_filename = "dc_scatter_plot.png"
    return render_template("dc.html", img_filename=scatterplot_filename, current_page='Analysis')


@app.route("/cea", methods=['POST'])
def cea():
    """
    Interface with dataset_consistency class, get all needed output that you would like to put for
    CEA metric on the webpage.
    """
    print("here cea")
    cea_scores = informer_obj.confidence_explanation_agreement(instance=True, visual=False)

    # random sample of 10 instances and their cea scores
    sample = random.sample(cea_scores, 10)

    # display histogram of cea scores for the input dataset
    hist = Informer.summary_statistics_from_cea(cea_scores)
    _, filename = hist.chart(8, low=-4, high=4, title="DISTRIBUTION OF CEA SCORES OVER THE ENTIRE DATASET", ui=True)

    # showcase # of instances that have cea score of 0 and the actual sequences

    examples_cea_zero = [x for x in cea_scores if x['CEA score'] == 0]
    examples_cea_zero = sorted(examples_cea_zero, key=lambda x: x['model confidence'], reverse=True)
    examples_cea_zero = [x['sentence(s)'] for x in examples_cea_zero]
    return render_template("cea.html", img_filename=filename, random_sample=sample, cea_zero=examples_cea_zero,
                           current_page='Analysis')


@app.route("/ci", methods=['POST'])
def ci():
    """
    Interface with dataset_consistency class, get all needed output that you would like to put for
    CI metric on the webpage. Currently, might need more infrastructural changes to get
    graphics. Only linear regression scores will be output.
    """
    pass


if __name__ == "__main__":
    app.run(debug=True)
