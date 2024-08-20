package morphology.morph

import morphology.data.DataStats
import morphology.ranking.{RankMorpheme, RankSequence, RankWord}
import zemberek.morphology.TurkishMorphology
import zemberek.morphology.analysis.SingleAnalysis.MorphemeData
import zemberek.morphology.analysis.{SingleAnalysis, WordAnalysis}
import zemberek.morphology.morphotactics.Morpheme

import java.util.Locale
import scala.collection.mutable
import scala.collection.parallel.CollectionConverters.ImmutableIterableIsParallelizable

class MorphAnalyzer {

  lazy val morphology = TurkishMorphology.createWithDefaults()
  val trLocale = new Locale("tr")

  var symbols = "!?+-*/-_{}[]()%^&#$@~|\\;,.'\" ".toCharArray.toSet
  var chars = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXabcçdefgğhiıjklmnoöprsştuüvyzxwq".toCharArray.toSet
  var nums = "1234567890".toCharArray.toSet
  val splitBy = "[\\+\\|\\→\\-\\>]"
  val wordBoundary = ">>>"
  var ambiguityFreq: Int = 1

  var doSkip: (Array[String] => Array[String]) = skipStem


  def setAmbiguityFreq(freq: Int): this.type = {
    ambiguityFreq = freq
    this
  }




  protected def anyof(character: Char, str: Set[Char]): Boolean = {
    str.contains(character)
  }

  protected def anyof(character: Char): Boolean = {
    anyof(character, symbols) || anyof(character, chars) || anyof(character, nums)
  }

  protected def convert(token: String): String = {
    token.toCharArray.filter(anyof(_)).mkString("")
  }

  protected def replace(id: String): String = {
    if ("Inf2".equals(id)) "Inf"
    else if("NarrPart".equals(id)) "Narr"
    else if("AorPart".equals(id)) "Aor"
    else id
  }

  protected def replace(analysis: SingleAnalysis): String = {
    val dataList = analysis.getMorphemes.toArray[Morpheme](Array[Morpheme]())
    var longSequence = mutable.Stack[String]()
    var preDB = false
    var prePOS = false

    dataList.foreach(morphemeData => {
      if (!morphemeData.derivational && morphemeData.pos == null) {
        longSequence.push(replace(morphemeData.id))
        prePOS = false
        preDB = false
      }

      else if (morphemeData.pos != null && preDB) {
        val lastID = longSequence.pop()
        val crrID = replace(morphemeData.id)
        longSequence.push(crrID)
        longSequence.push(lastID)
        prePOS = true
        preDB = false
      }

      else if (morphemeData.derivational) {
        longSequence.push(replace(morphemeData.id))
        preDB = true
        prePOS = false
      }
      else if (morphemeData.pos != null) {
        prePOS = true
        preDB = false
        val crrID = replace(morphemeData.id)
        longSequence.push(crrID)
      }
    })

    val reverseStack = new mutable.Stack[String]()
    prePOS = false
    while (longSequence.nonEmpty) {
      val item = longSequence.pop()
      if (isPos(item) && prePOS) {
        prePOS = true
      }
      else if (isPos(item)) {
        reverseStack.push(item)
        prePOS = true
      }
      else {
        reverseStack.push(item)
        prePOS = false
      }
    }

    reverseStack.toSeq.mkString("+")

  }

  protected def isPos(item: String): Boolean = {
    ("Noun".equals(item)) || "Verb".equals(item) || "Adj".equals(item) || "Adv".equals("item") || "Det".equals(item)
  }

  protected def analyze(morphology: TurkishMorphology, token: String): Array[String] = {

    var result = Array[String]()
    val iter = morphology.analyze(token).iterator()

    while (iter.hasNext) {
      val single = iter.next()
      val item = single.getStem + "+" + replace(single)
      result = result :+ item
    }

    if (result.isEmpty && token.nonEmpty && token(0).isLower) {
      result = result :+ ("UNK" + "+Noun+A3sg")
    }
    else if (result.isEmpty && token.isEmpty) {
      result = result :+ ("UNK" + "+Noun+A3sg")
    }

    result
  }

  protected def analyzeSentence(morphology: TurkishMorphology, sentence: String): Array[Array[String]] = {
    val analysis = morphology.analyzeSentence(sentence).toArray[WordAnalysis](Array[WordAnalysis]())
    analysis.map(wordAnalysis => {
      val analysis = wordAnalysis.getAnalysisResults.toArray[SingleAnalysis](Array[SingleAnalysis]())

      if (analysis.nonEmpty) {
        analysis.map(singleAnalysis => wordAnalysis.getNormalizedInput() + "|" + singleAnalysis.getStem.toLowerCase(trLocale) + "+" + replace(singleAnalysis)
          .toLowerCase(trLocale))
      }
      else {
        Array(wordAnalysis.getNormalizedInput() + "|" + "unk+noun+a3sg")
      }
    })
  }

  protected def skipStem(wordAnalyze: Array[String]): Array[String] = {
    wordAnalyze.map(analyze => analyze.split("\\+"))
      .map(analyze => analyze.slice(1, analyze.length).mkString("+"))
  }

  protected def noSkipStem(wordAnalyze: Array[String]): Array[String] = {
    wordAnalyze.map(analyze => analyze.split("\\+"))
      .map(analyze => analyze.slice(0, analyze.length).mkString("+"))
  }

  def parseToken(token: String): Array[String] = {
    analyze(morphology, convert(token))
  }

  def parse(tokens: Array[String]): Array[Array[String]] = {
    tokens.map(token => analyze(morphology, convert(token)))
  }

  def parseSentence(sentence: String): Array[Array[String]] = {
    analyzeSentence(morphology, sentence)
  }

  def parseSkip(tokens: Array[String]): Array[Array[String]] = {
    tokens.map(token => doSkip(analyze(morphology, convert(token))))
  }

  def flatten(analysis: Array[String]): Array[String] = {
    analysis.flatMap(analyze => wordBoundary +: analyze.split(splitBy))
  }

  def countAnalysis(array: Array[Array[String]]): Long = {
    val total = array.map(_.length).foldLeft[Long](1L) { case (a, b) => a * b }
    total
  }


  def combinatoric(input: Array[Array[String]]): Array[Array[String]] = {
    val result = Array[Array[String]](Array())
    val totalCombinations = countAnalysis(input)
    if (totalCombinations < ambiguityFreq) combinatoric(input, result)
    else result
  }

  def combinatoricMorpheme(input: Array[RankWord]): Array[RankSequence] = {
    val morphemes = input.map(_.rankMorphemes)
    val result = Array[Array[RankMorpheme]](Array())
    val combinations = combinatoricMorpheme(morphemes, result)
    combinations.map(path=> RankSequence(path, 0))
  }

  protected def combinatoric(input: Array[Array[String]], result: Array[Array[String]], i: Int = 0): Array[Array[String]] = {

    if (i == input.length) result
    else {

      val crr = i;
      var array = Array[Array[String]]()

      for (k <- 0 until input(crr).length) {
        Range(0, result.length).par.map(i => {
          result(i) :+ input(crr)(k)
        }).toArray.foreach(current => {
          array = array :+ current
        })
      }

      combinatoric(input, array, crr + 1)

    }
  }

  protected def combinatoricMorpheme(input: Array[Array[RankMorpheme]], result: Array[Array[RankMorpheme]], i: Int = 0): Array[Array[RankMorpheme]] = {

    if (i == input.length) result
    else {

      val crr = i;
      var array = Array[Array[RankMorpheme]]()

      for (k <- 0 until input(crr).length) {
        Range(0, result.length).par.map(i => {
          result(i) :+ input(crr)(k)
        }).toArray.foreach(current => {
          array = array :+ current
        })
      }

      combinatoricMorpheme(input, array, crr + 1)

    }
  }
}
