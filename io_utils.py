import logging
from os import path
from os.path import dirname, join
from arekit.common.utils import create_dir_if_not_exists
from arekit.contrib.experiments.cv.default import SimpleCVFolding
from arekit.contrib.experiments.cv.doc_stat.rusentrel import RuSentRelDocStatGenerator
from arekit.contrib.experiments.cv.sentence_based import SentenceBasedCVFolding
from arekit.contrib.experiments.data_io import DataIO
from arekit.contrib.experiments.neutral.annot.rusentrel_three_scale import RuSentRelThreeScaleNeutralAnnotator
from arekit.contrib.experiments.neutral.annot.rusentrel_two_scale import RuSentRelTwoScaleNeutralAnnotator
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.source.embeddings.rusvectores import RusvectoresEmbedding
from arekit.source.rusentrel.opinions.formatter import RuSentRelOpinionCollectionFormatter
from arekit.source.rusentrel.synonyms import RuSentRelSynonymsCollection

logger = logging.getLogger(__name__)


class RuSentRelBasedExperimentsIOUtils(DataIO):

    def __init__(self, init_word_embedding=True):
        self.__stemmer = MystemWrapper()
        self.__synonym_collection = RuSentRelSynonymsCollection.load_collection(
            stemmer=self.__stemmer,
            is_read_only=True)
        self.__opinion_formatter = RuSentRelOpinionCollectionFormatter
        self.__neutral_annotator = self.__init_two_scale_neutral_annotator()
        self.__word_embedding = self.__create_word_embedding() if init_word_embedding else None
        self.__cv_folding_algorithm = self.__init_sentence_based_cv_folding_algorithm()

    # region public properties

    @property
    def Stemmer(self):
        return self.__stemmer

    @property
    def SynonymsCollection(self):
        return self.__synonym_collection

    @property
    def NeutralAnnontator(self):
        return self.__neutral_annotator

    @property
    def WordEmbedding(self):
        return self.__word_embedding

    @property
    def OpinionFormatter(self):
        return self.__opinion_formatter

    @property
    def CVFoldingAlgorithm(self):
        return self.__cv_folding_algorithm

    # endregion

    # region private methods

    def __create_word_embedding(self):
        we_filepath = path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")
        logger.info("Loading word embedding: {}".format(we_filepath))
        return RusvectoresEmbedding.from_word2vec_format(filepath=we_filepath,
                                                         binary=True)

    def __init_sentence_based_cv_folding_algorithm(self):
        return SentenceBasedCVFolding(
            docs_stat=RuSentRelDocStatGenerator(synonyms=self.__synonym_collection),
            docs_stat_filepath=path.join(self.get_data_root(), u"docs_stat.txt"))

    def __init_simple_cv_folding_algoritm(self):
        return SimpleCVFolding()

    def __init_two_scale_neutral_annotator(self):
        return RuSentRelTwoScaleNeutralAnnotator(data_io=self)

    def __init_three_scale_neutral_annotator(self):
        return RuSentRelThreeScaleNeutralAnnotator(data_io=self,
                                                   stemmer=self.__stemmer)

    # endregion

    # region public methods

    def get_data_root(self):
        return path.join(dirname(__file__), u"data/")

    def get_experiments_dir(self):
        experiments_name = u'rusentrel'
        target_dir = join(self.get_data_root(), u"./{}/".format(experiments_name))
        create_dir_if_not_exists(target_dir)
        return target_dir

    def get_word_embedding_filepath(self):
        return path.join(self.get_data_root(), u"w2v/news_rusvectores2.bin.gz")

    # endregion
