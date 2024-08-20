package morphology.morph

import morphology.data.Params
import nn.embeddings.EmbedParams
import utils.FreqWordTokenizer

import java.io.{File, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.util.regex.Pattern
import scala.collection.parallel.CollectionConverters.{ArrayIsParallelizable, ImmutableIterableIsParallelizable}
import scala.io.Source
import scala.util.Random
import scala.util.control.Breaks

class PTBTokenizer(val filename: String = "/resources/binary/dictionary.bin") extends Serializable {

  val binFilename = new File("").getAbsoluteFile().getAbsolutePath + filename
  val regex1 = "([abcçdefgğhıijklmnoöprsştuüvyzwqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWX\\p{L}]+)"
  val regex2 = "(\\d+)"
  val regex3 = "(\\p{Punct}+)"
  val regex4 = "([\\'\\>\\<\\-\\+\\*\\/\\$\\#\\~\\{\\}\\(\\)\\[\\]\\%\\@\\|\\;\\:\"\\.]+)"
  val regexArray = Array(regex1, regex2, regex3, regex4)
  val patternArray = regexArray.map(Pattern.compile(_, Pattern.DOTALL))
  val sentenceFilename = "resources/text/sentences-tr.txt";
  var frequency = Map[String, Double]()
  val rnd = new Random()

  def freqConstruct(filename: String): PTBTokenizer = {

    Source.fromFile(filename).getLines().foreach(sentence => {
      val tokens = standardTokenizer(sentence).filter(item => item.matches(regex1))
      val sentences = Array(tokens.mkString(" "))
      sentences.foreach(sentence => {
        val splited = sentence.split("\\s+")
        Range(1, 3).toArray.flatMap(s => splited.sliding(s).map(items => items.mkString(" ")).toArray)
          .foreach(item => {
            frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + 1d)
          })
      })
    })

    frequency = frequency.filter { case (item, count) => count >= 2 }
    frequency.filter { case (item, count) => {

      val prb = -Math.log(1E-10 + count / frequency.size)
      val total = item.split("\\s+")
        .map(subitem => -Math.log(1E-10 + frequency.getOrElse(subitem, 0d) / frequency.size))
        .sum
      prb / total >= 1.0

    }
    }

    save()
  }


  def freqNGramConstruct(filename: String, rsize: Int = 100000): PTBTokenizer = {
    val rndSet = Range(0, rsize).map(_ => rnd.nextInt(100000000)).toSet
    var index = 0
    var count = 0
    val breaks = new Breaks()
    breaks.breakable {
      Source.fromFile(filename).getLines().foreach(sentence => {
        if (rndSet.contains(index) && sentence.length < 120) {
          println("Processing: " + sentence)
          val tokens = standardTokenizer(sentence).filter(item => item.matches("[a-zA-ZğüşıçöĞÜŞİÇÖ]+"))
          val sentences = ngramCombinations(tokens)
          sentences.foreach(sentence => {
            val splited = sentence.split("\\s+")
            Range(1, 3).toArray
              .flatMap(s => splited.sliding(s).toArray.map(items => items.mkString(" ")))
              .foreach(item => {
                frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + 1d)
              })
          })

          count += 1
        }
        if (count >= rsize) breaks.break()
        index += 1
      })
    }


    save()

  }

  def prune(): this.type = {

    frequency = frequency.filter { case (item, count) => count >= 2 }
    frequency = frequency.filter { case (item, count) => {
      val prb = -Math.log(1E-10 + count / frequency.size)
      val total = item.split("\\s+")
        .map(subitem => -Math.log(1E-10 + frequency.getOrElse(subitem, 0d) / frequency.size))
        .sum
      prb / total >= 1.0

    }
    }

    this
  }

  def save(): PTBTokenizer = {
    val outputStream = new ObjectOutputStream(new FileOutputStream(binFilename))
    val array = frequency.toArray
    outputStream.writeInt(array.size)

    for (i <- 0 until array.size) {
      val item = array(i)
      outputStream.writeObject(item)
    }

    outputStream.close()
    this
  }

  def load(): PTBTokenizer = {
    if (new File(binFilename).exists()) {
      val inputStream = new ObjectInputStream(new FileInputStream(binFilename))
      val size = inputStream.readInt()
      var array = Array[(String, Double)]()

      for (i <- 0 until size) {
        val item = inputStream.readObject().asInstanceOf[(String, Double)]
        array = array :+ item
      }


      inputStream.close()
      this.frequency = array.toMap
      this
    }
    else {
      this
    }
  }

  def merge(freqWordTokenizer: PTBTokenizer): PTBTokenizer = {
    freqWordTokenizer.frequency.foreach { case (item, value) => {
      frequency = frequency.updated(item, frequency.getOrElse(item, 0d) + value)
    }
    }

    this
  }


  def freqTokenizer(sentence: String): Array[String] = {

    val tokens = standardTokenizer(sentence)
    var result = Array[String]()
    var i = 0;
    while (i < tokens.length) {
      var f = i
      var b = true
      val break = new Breaks()

      break.breakable {
        val mx = Math.min(i + 3, tokens.length)
        for (j <- mx until i + 1 by -1) {
          val slice = tokens.slice(i, j).mkString(" ")
          if (frequency.contains(slice)) {
            result = result :+ slice
            f = j
            b = false
            break.break()
          }
        }
      }

      if (b) result = result :+ tokens(i)
      i = f + 1
    }

    result
  }

  def freqNGramTokenizer(sentence: String): Array[String] = {

    val tokens = standardTokenizer(sentence)
    val sentences = ngramCombinations(tokens)

    val finalResult = sentences.map(sentence => {
      var result = Array[(String, Double)]()
      val items = sentence.split("\\s+")
      var i = 0;
      while (i < tokens.length) {
        var f = i
        var b = true
        val break = new Breaks()

        break.breakable {
          val mx = Math.min(i + 3, tokens.length)
          for (j <- mx until i + 1 by -1) {
            val slice = tokens.slice(i, j).mkString(" ")
            if (frequency.contains(slice)) {
              result = result :+ (slice, frequency(slice))
              f = j
              b = false
              break.break()
            }
          }
        }

        if (b) {
          val d = frequency.getOrElse(tokens(i), 0d)
          result = result :+ (tokens(i), d)
        }
        i = f + 1
      }

      result
    })

    finalResult.map(array => (array, array.map(_._2).sum / array.length))
      .sortBy(_._2)
      .last._1.map(_._1)

  }

  //separate everything
  def standardTokenizer(sentence: String, pattern: Pattern, start: Int): Option[(String, Int, Int)] = {

    val matcher = pattern.matcher(sentence)
    if (matcher.find(start)) {
      val (s, e) = (matcher.start(), matcher.end())
      val group = sentence.substring(s, Math.min(sentence.length, Math.max(e, s + 1))).trim
      if (group.nonEmpty) {
        return Some((group, s, e))
      }
    }

    None
  }

  def standardTokenizer(sentence: String, start: Int): Option[(String, Int, Int)] = {

    val res = patternArray.par.flatMap(p => standardTokenizer(sentence, p, start)).toArray.sortBy(tuple => tuple._3 - tuple._2)
      .reverse
    if (res.isEmpty) None
    else Some(res.head)

  }

  def standardTokenizer(sentence: String): Array[String] = {
    var start = 0
    var found = standardTokenizer(sentence, start)
    var array = Array[String]()

    while (found.isDefined) {
      val (group, start, end) = found.get
      array = array :+ group
      found = standardTokenizer(sentence, end)
    }

    array
  }

  protected def combinatoric(input: Array[Array[String]], result: Array[Array[String]], i: Int = 0): Array[Array[String]] = {

    if (i == input.length) result
    else {

      var crr = i;
      var array = Array[Array[String]]()

      for (k <- 0 until input(crr).length) {
        for (i <- 0 until result.length) {
          val current = result(i) :+ input(crr)(k)
          array = array :+ current
        }
      }

      combinatoric(input, array, crr + 1)
    }
  }

  def ngramCombinations(sentence: String): Array[String] = {
    //find the root
    val arrays = standardTokenizer(sentence).map(token => token.toSeq.sliding(5, 1).map(_.unwrap).toArray)
    combinatoric(arrays, Array.empty[Array[String]]).map(tokens => tokens.mkString(" "))
  }


  def ngramCombinations(sentence: Array[String]): Array[String] = {
    //find the root
    val arrays = sentence.map(token => {
      val min = Math.min(token.length, 5)
      (token.slice(0, min) +: token.substring(min).toSeq.sliding(2, 2).toArray.map(_.unwrap))
    })
    combinatoric(arrays, Array(Array[String]())).map(tokens => tokens.mkString(" "))
  }

}

object PTBTokenizer {
  val params = new Params()

  def construct(): Unit = {
    val range = Range(0, params)
    range.toArray.sliding(8, 8).foreach { arr => {

      val result = arr.toList.par.map(i => {
        val tokenizer = new PTBTokenizer()
        tokenizer.load()
        tokenizer.freqNGramConstruct(params.sentenceFilename, 100000)
        tokenizer
      }).foldRight[PTBTokenizer](new PTBTokenizer()) { case (a, main) => main.merge(a) }


      result
        .prune()
        .save()

    }
    }
  }

  def main(args: Array[String]): Unit = {
    construct()
  }

}