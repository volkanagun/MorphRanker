package morphology.data

import morphology.morph.{MorphAnalyzer, Tokenizer}

import java.io.{File, FileWriter, PrintWriter}
import java.text.BreakIterator
import java.util.Locale
import scala.collection.immutable.Map
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.io.Source

object Dataset {

  val locale = new Locale("tr")
  val analyzer = new MorphAnalyzer()
  val tokenizer = new Tokenizer()

  def parseSentence(sentenceLine: String): Sentence = {
    val tokens = tokenizer.standardTokenizer(sentenceLine)
    val words = analyzer.parseSkip(tokens).zip(tokens).zipWithIndex.map { case ((analysis, token), index) => {
      Word(index, token, analysis)
    }
    }

    Sentence(sentenceLine.hashCode, sentenceLine).setWords(words)
  }

  def parseSentence(analyzer: MorphAnalyzer, sentenceLine: String): Sentence = {

    val words = analyzer.parseSentence(sentenceLine).zipWithIndex.map { wordAnalysisPair => {
      val wordIndex = wordAnalysisPair._2
      val wordAnalysis = wordAnalysisPair._1
      val token = wordAnalysis.flatMap(item => item.split("\\|")).head
        .toLowerCase(locale)
      val morphemes = wordAnalysis.map(item => item.split("\\|").last)
        .map(_.toLowerCase(locale))
      Word(wordIndex, token, morphemes)
    }
    }

    Sentence(sentenceLine.hashCode, sentenceLine)
      .setWords(words)
  }

  def replace(analysis: String): String = {
    analysis
      .replaceAll("([\\p{L}\\d]+)\\+pos", "$1")
      .replaceAll("\\^db", "")
      .replaceAll("(postp\\+(\\p{L}+))", "postp")
      .replaceAll("(\\p{L}+)\\+(pnon)", "$1")
      .replaceAll("(\\p{L}+)\\+(nom)", "$1")
      .replaceAll("(\\p{L}+)\\+(prop)", "$1")
      .replaceAll("adverb", "adv")
      .replaceAll("num\\+card", "num")
      .replaceAll("ınf2", "ınf")
  }

  def readSentences(filenames: Array[String]): Array[Sentence] = {
    filenames.par.flatMap(filename => readSentences(filename)).toArray
  }

  def readSentences(filename: String): Array[Sentence] = {
    var sentences = Array[Sentence]()
    var id = 0
    var index = 1
    var sentence: Sentence = null
    var words = Array[Word]()
    Source.fromFile(filename, "UTF-8").getLines().foreach(line => {

      if (line.trim.startsWith("<S")) {
        sentence = Sentence(id, line)
        index = 1
      }
      else if (line.trim.startsWith("</S>")) {
        sentences = sentences :+ sentence.setWords(words)
        words = Array()
        id += 1
      }
      else if (!(line.startsWith("<DOC>") || line.startsWith("<TITLE>") || line.startsWith("</TITLE>") || line.startsWith("</DOC>"))) {
        val lowered = line.toLowerCase(locale)
        val token = lowered.split("\\s+")
        val word = token.head
        val analyze = token.tail.map(replace)
        if (sentence != null) words = words :+ Word(index, word, analyze)
        else println("Null item: " + line)
        index += 1
      }
    })

    sentences
  }

  def aggregateLabel(filename: String): Map[String, Label] = {
    val sentenceList = readSentences(filename)
    sentenceList.flatMap(sentence => sentence.toNonStemLabels())
      .groupBy(_.tags).view.mapValues(array => {
        array.reduceRight[Label] { case (a, main) => main.mege(a) }
      }).toMap
  }

  def aggregateLabel(filenames: Array[String], sliceSize: Int): Map[String, Label] = {
    val sentenceList = readSentences(filenames)
    sentenceList.par.flatMap(sentence => sentence.toNonStemLabels(sliceSize).filter(_.trueLabel))
      .toArray
      .groupBy(_.tags).view.mapValues(array => {
        array.reduceRight[Label] { case (a, main) => main.mege(a) }
      }).toMap
  }

  def aggregateUniqueLabel(filename: String, sliceWindow: Int): Map[String, Label] = {
    val sentenceList = readSentences(filename)
    sentenceList.flatMap(sentence => sentence.toUniqueLabels(sliceWindow))
      .groupBy(_.tags).view.mapValues(array => {
        array.reduceRight[Label] { case (a, main) => main.mege(a) }
      }).toMap
  }

  def splitSentences(text: String): Array[String] = {
    var sentenceArray = Array[String]()
    val iterator = BreakIterator.getSentenceInstance(locale)
    iterator.setText(text)
    var previous = 0
    var current = iterator.next()
    while (current != BreakIterator.DONE) {
      sentenceArray = sentenceArray :+ text.substring(previous, current)
      previous = current
      current = iterator.next()
    }

    sentenceArray
  }


  def nonAmbiguousSelecting(ambiguousLabelSet: Set[Label], trainFilename: String,
                            ngramSliceSize: Int, startIndex: Int, searchSize: Int): Map[Label, Array[Sentence]] = {

    val crrAnalyzer = new MorphAnalyzer()
    val sampleMap = Source.fromFile(trainFilename).getLines().zipWithIndex.filter(_._2 >= startIndex).take(searchSize)
      .map(_._1).toArray
      .par
      .flatMap(pair => {
        splitSentences(pair)
      }).flatMap(sentenceLine => {
        val sentence = parseSentence(crrAnalyzer, sentenceLine)
        val sentenceLabels = sentence.toNonStemNotAmbiguousLabels(ngramSliceSize)
        val filteredLabels = sentenceLabels.filter(label => ambiguousLabelSet.contains(label))
        filteredLabels.map(label => (label, sentence))
      }).groupBy(_._1)
      .map(pair=> (pair._1 -> pair._2.map(_._2).toArray))
      .toArray
      .toMap

    sampleMap

  }

  def ambiguousLabels(testFilename: String, minimumThreshold: Double): Set[Label] = {
    val ambiguousLabels = aggregateLabel(testFilename).filter { case (_, label) => label.ambiguousScore() > minimumThreshold }.toArray
      .sortBy(_._2.ambiguousScore())
      .reverse.map(_._2).toSet
    ambiguousLabels
  }

  def ambiguousLabels(testFilenames: Array[String], sliceSize: Int, minimumThreshold: Double): Set[Label] = {
    val ambiguousLabels = aggregateLabel(testFilenames, sliceSize).filter { case (_, label) => label.ambiguousScore() > minimumThreshold }.toArray
      .sortBy(_._2.ambiguousScore())
      .reverse.map(_._2).toSet
    ambiguousLabels
  }

  def ambiguousUniqueLabels(testFilename: String, minimumThreshold: Double, sliceWindow: Int): Set[Label] = {
    val ambiguousLabels = aggregateUniqueLabel(testFilename, sliceWindow).filter { case (_, label) => label.ambiguousScore() > minimumThreshold }.toArray
      .sortBy(_._2.ambiguousScore())
      .reverse.map(_._2).toSet
    ambiguousLabels
  }

  def nonAmbiguousLabels(testFilename: String): Set[Label] = {
    val ambiguousLabels = aggregateLabel(testFilename).filter { case (_, label) => label.ambiguousScore() == 1.0 }.toArray
      .reverse.map(_._2).toSet
    ambiguousLabels
  }

  def allTimeAmbiguousLabels(testFilename: String, threshold: Double): Set[Label] = {
    val ambiguousLabelSet = ambiguousLabels(testFilename, threshold)
    val sentences = readSentences(testFilename)
    val notAllAmbiguous = sentences.map(sentence => {
        (sentence, sentence.toNonStemLabels())
      }).filter { case (_, labels) => {
        val notAmbiguous = labels.filter(label => label.ambiguousScore() == 1.0)
          .filter(label => ambiguousLabelSet.contains(label))
        notAmbiguous.nonEmpty
      }
      }.flatMap { case (sentence, labels) => {
        labels.map(label => (label, sentence))
      }
      }.groupBy(_._1).view.mapValues(_.map(_._2))
      .toMap


    val allAmbiguous = ambiguousLabelSet.filter(label => !notAllAmbiguous.contains(label))
    allAmbiguous
  }

  def nonAmbiguousSamples(testFilename: String, threshold: Double): Map[Label, Array[Sentence]] = {
    val ambiguousLabelSet = ambiguousLabels(testFilename, threshold)
    val sentences = readSentences(testFilename)
    val notAllAmbiguous = sentences.map(sentence => {
        (sentence, sentence.toNonStemLabels())
      }).filter { case (_, labels) => {
        val notAmbiguous = labels.filter(label => label.ambiguousScore() == 1.0)
          .filter(label => ambiguousLabelSet.contains(label))
        notAmbiguous.nonEmpty
      }
      }.flatMap { case (sentence, labels) => {
        labels.map(label => (label, sentence))
      }
      }.groupBy(_._1).view.mapValues(_.map(_._2))
      .toMap

    notAllAmbiguous
  }

  def rankAmbiguousSlices(testFilename: String, threshold: Double, sliceWindow: Int): Map[Label, Double] = {
    val allAmbiguousSet = allTimeAmbiguousLabels(testFilename, threshold)
    val allAmbiguosSlices = allAmbiguousSet.map(ambiguousLabel => (ambiguousLabel, ambiguousLabel.slice(sliceWindow)))
      .toMap
    val sentences = readSentences(testFilename)
    val allSlices = sentences.flatMap(sentence => sentence.toNonStemLabels().flatMap(label => label.slice(sliceWindow)))
      .filter(_.ambiguousScore() == 1.0).toSet

    allAmbiguosSlices.view.mapValues(slices => {
      slices.filter(slice => allSlices.contains(slice)).length.toDouble / slices.length
    }).toMap

  }

  def nonAmbiguousTestSamples(testFilename: String, threshold: Double, sliceWindow: Int): Map[Label, Array[Sentence]] = {
    val allAmbiguousSet = allTimeAmbiguousLabels(testFilename, threshold)
    val allAmbiguosSlices = allAmbiguousSet.flatMap(ambiguousLabel => ambiguousLabel.slice(sliceWindow))

    val sentences = readSentences(testFilename)
    val allNonAmbiguousSamples = sentences.flatMap(sentence => sentence.toNonStemLabels()
      .flatMap(label => label.slice(sliceWindow))
      .filter(label => {
        label.ambiguousScore() == 1.0 && allAmbiguosSlices.contains(label)
      }).map(label => {
        (label, sentence)
      }))

    allNonAmbiguousSamples.groupBy(_._1).view.mapValues(array => array.map(_._2)).toMap ++ nonAmbiguousSamples(testFilename, threshold)
  }

  def labelLine(label: Label, sentence: Sentence): String = {
    label.index + "\t\t" + label.tags + "\t\t" + sentence.text
  }

  def ambiguousWords(trainFilename: String, testFilename: String, labelFilename: String, ambiguityThreshold: Double, sliceWindow: Int): Unit = {
    val mainAmbiguousMap = ambiguousUniqueLabels(testFilename, ambiguityThreshold, sliceWindow).groupBy(_.toAnalysisLabel())
    val searcSize = 10000000
    val threadSize = 5000
    val sentenceLength = 200
    val pw = new PrintWriter(labelFilename)
    val labelSet = Source.fromFile(trainFilename).getLines().filter(_.length < sentenceLength)
      .take(searcSize).sliding(threadSize, threadSize).zipWithIndex
      .toArray.par.flatMap(pairBatches => {
        println("Current index: " + pairBatches._2 + "/" + searcSize / threadSize);
        val crrAnalyzer = new MorphAnalyzer()
        val lines = pairBatches._1.toArray
        val labels = lines.flatMap(line => {
          parseSentence(crrAnalyzer, line).toNonStemLabels().filter(_.notAmbiguous())
            .flatMap(label => label.slice(sliceWindow))
        })
        labels
      }).toArray.toSet

    println("Ambiguos size: " + mainAmbiguousMap.size)
    var count = 0;
    mainAmbiguousMap.keys.toArray.par.map(crrMainAnalysis => {
      val diffLabels = mainAmbiguousMap(crrMainAnalysis)
      println("Current label: " + crrMainAnalysis.analysis)
      val hasSample = diffLabels.exists(diffLabel => labelSet.contains(diffLabel))
      (hasSample, crrMainAnalysis)
    }).toArray.foreach { case (hasSample, ambiguousLabel) => {
      if (!hasSample) {
        println("Ambiguous")
        pw.println(ambiguousLabel.word + ":" + ambiguousLabel.analysis)
      }
      else {
        count += 1
        println("Not ambiguous...")
      }
    }
    }

    println("Not ambiguous rate = " + count.toDouble / mainAmbiguousMap.size)
    pw.close()
  }

  def analyzeAmbiguous(testFilename: String): Unit = {
    Source.fromFile(testFilename).getLines().toArray.reverse.foreach(line => {
      val Array(token, analysis) = line.split("\\:")
      println("Token: " + token)
      println("Analysis: " + analysis)
      println(analyzer.parseToken(token).mkString("\n"))
    })
  }

  def mergeSentences(sentenceFolder: String, destinationFilename: String): Unit = {

    val files = new File(sentenceFolder).listFiles()
    val pw = new PrintWriter(destinationFilename)
    files.foreach(f => Source.fromFile(f).getLines().foreach(line => {
      pw.println(line)
    }))

    pw.close()
  }

  def constructDataset(testFilenames: Array[String], trainFilename: String, targetFilename: String,
                       ambiguousThreshold: Double,
                       maxEpocs: Int,
                       sliceSize: Int,
                       searchSize: Int,
                       minLabelSamples: Int,
                       maxLabelSamples: Int): Unit = {

    var mergeLabelMap = Map[Label, Array[Sentence]]()
    var ambiguousLabelSet = ambiguousLabels(testFilenames, sliceSize, ambiguousThreshold)

    var sampleSet = Set[Sentence]()

    var crrIndex = 0
    var sampleSize = 0
    var nthreads = 24
    Range(0, maxEpocs).sliding(nthreads, nthreads).foreach(rangeSeq=>{
      rangeSeq.par.map(crrIndex=>{
        println("Epoc: " + crrIndex)
        val startIndex = crrIndex * searchSize
        nonAmbiguousSelecting(ambiguousLabelSet, trainFilename, sliceSize, startIndex, searchSize)
      }).toArray.foreach(newLabelMap=>{
        newLabelMap.foreach { case (label, array) => {
          mergeLabelMap = mergeLabelMap.updated(label, mergeLabelMap.getOrElse(label, Array()) ++ array)
        }}
      })

      ambiguousLabelSet = mergeLabelMap.filter(_._2.length < minLabelSamples).map(_._1).toSet
      sampleSet = mergeLabelMap.toArray.flatMap(pair => pair._2.take(maxLabelSamples)).toSet
      val counts = ambiguousLabelSet.map(label=> label.toString +":"+mergeLabelMap(label).size).mkString("[",",","]")
      println("Sample size: " + sampleSet.size)
      println("Search label size: " + ambiguousLabelSet.size)
      println("Label counts: " + counts)
      crrIndex = crrIndex + 1
      sampleSize = sampleSet.size
    })

    mergeLabelMap = mergeLabelMap.view.mapValues(sentences => sentences.sortBy(_.averageAmbiguity()).take(maxLabelSamples)).toMap
    val pw = new PrintWriter(targetFilename)
    sampleSet.foreach { sentence => {
      pw.println(sentence.text)
    }
    }

    pw.close()
  }

  def ambiguousCounts(): Unit = {
    val testFilenames = new Params().vocabFilenames
    testFilenames.foreach(testFilename=>{
      val sentenceList = readSentences(testFilename)
      val totalWords = sentenceList.map(sentence=> sentence.words.length).sum
      val totalAmbiguous = sentenceList.map(sentence=> sentence.words.filter(item => !item.notAmbiguous()).length).sum
      println("Test filename: "+testFilename + " Ambiguous Count: "+totalAmbiguous + " Total Tokens: "+totalWords)
    })
  }

  def main(args: Array[String]): Unit = {

    val testFilenames = new Params().vocabFilenames
    val sentenceFolder = new Params().sentenceFolder
    val sentenceFilename = new Params().trainingFilename
    //val targetFilename = new Params().ambiguousLabelFilename
    val targetFilename = new Params().sentenceLabel4Filename

    val maxEpocs = 5000
    val slice = 3
    val threshold = 0.0
    val minSamples = 500
    val maxSamples = 500
    val searchSize = 5000
    //ambiguousWords(trainFilename, testFilename, targetFilename, threshold, slice)
    //constructDataset(testFilenames, sentenceFilename, targetFilename, threshold, maxEpocs, slice, searchSize, minSamples, maxSamples)
    ambiguousCounts()
  }
}
