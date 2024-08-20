package morphology.data

case class Sentence(id: Int, var text: String, var words: Array[Word] = Array()) {

  def adjust(): this.type = {
    val wordStart = Word(0, "START", Array("START+START"))
    val wordEnd = Word(words.length + 1, "END", Array("END+END"))

    words = wordStart +: words
    words = words :+ wordEnd
    this
  }

  override def hashCode(): Int = text.hashCode()

  override def equals(obj: Any): Boolean = {
    obj.asInstanceOf[Sentence].text.equals(text)
  }

  def add(word: Word): this.type = {
    this.words = this.words :+ word
    this
  }

  def setWords(words: Array[Word]): this.type = {
    this.words = words
    adjust()
  }

  def length(): Int = {
    words.length
  }


  def toNonStemLabels(): Array[Label] = {
    words.flatMap(word => word.toNonStemLabels())
  }

  def toNonStemLabels(slice: Int): Array[Label] = {
    words.flatMap(word => word.toNonStemLabels(slice))
  }

  def toNonStemNotAmbiguousLabels(slice: Int): Array[Label] = {
    words.filter(word=> word.notAmbiguous()).flatMap(word => word.toNonStemLabels(slice))
  }

  def toNonStemWordLabels(slice: Int): Array[Array[Label]] = {
    words.map(word => word.toNonStemLabels(slice))
  }

  def toNonStemTrueWordLabels(slice: Int): Array[Array[Label]] = {
    words.map(word => {
      if(word.notAmbiguous()) word.toNonStemLabels(slice)
      else Array()
    })
  }

  def toWithStemLabels(slice: Int): Array[Label] = {
    words.flatMap(word => word.toStemLabels(slice))
  }

  def toLabels(): Array[Label] = {
    words.flatMap(word => word.toDirectLabels())
  }

  def toWordLabels(): Array[Array[Label]] = {
    words.map(word => word.toNonStemLabels().toArray)
  }

  def toWordSliceLabels(sliceSize: Int): Array[Array[Array[Label]]] = {
    words.map(word => word.toStemLabels(sliceSize).map(_.slice(sliceSize)))
  }

  def toUniqueLabels(sliceSize: Int): Array[Label] = {
    words.flatMap(word => word.toUniqueSlices(sliceSize))
  }

  def toUniqueWordLabels(sliceSize: Int): Array[Array[Label]] = {
    words.map(word => word.toUniqueSlices(sliceSize).toArray)
  }

  def toWordLabels(sliceSize: Int): Array[Array[Label]] = {
    words.map(word => word.toStemLabels(sliceSize))
  }

  def toUniqueAnalysisLabels(sliceSize: Int): Array[Array[Array[Label]]] = {
    words.map(word => word.toUniqueAnalysisSlices(sliceSize))
  }

  def countAmbiguity(): Int = {
    words.map(word => word.originalAnalyses.length).foldRight[Int](1) {
      case (i, main) => i * main
    }
  }

  def averageAmbiguity(): Double = {
    val score = countAmbiguity()
    score.toDouble / words.length
  }

  def countAnalysis(): Int = {
    words.map(word => word.originalAnalyses.length).foldRight[Int](0) {
      case (i, main) => i + main
    }
  }

  def countTags(): Int = {
    words.map(word => word.originalAnalyses).foldRight[Int](0) {
      case (i, main) => i.flatMap(analysis => analysis.split("\\+")).length + main
    }
  }


  def toPairs(maxAmbiguity: Int, sliceSize: Int, func: (Int) => String, orders: Array[Int]): Array[(String, Label, Label)] = {

    orders.flatMap(order => {
      words.sliding(order, 1).toArray
        .flatMap(elem => {

          val distance = order - 1
          val linkName = "Forward-" + func(distance)
          val array = if (linkName.contains("Local")) {
            val allSeq = elem.head.toDistinctLabels(sliceSize)
            allSeq.flatMap(crrSeq=>{
              crrSeq.sliding(2, 1).map(localSeq=>{
                val headLabel = localSeq.head
                val lastLabel = localSeq.last
                (linkName, headLabel, lastLabel)
              })
            })

          }
          else {
            val aLemma = elem.head.toLemmaLabels()
            val bLemma = elem.last.toLemmaLabels()
            val aResult = aLemma ++ elem.head.toNonStemLabels(sliceSize)
            val bResult = bLemma ++ elem.last.toNonStemLabels(sliceSize)
            aResult.flatMap(aLabel => {
              bResult.map(bLabel => {
                (linkName, aLabel, bLabel)
              })
            })
          }

          val nonAmbiguous = array.filter(pair => {
            val ambiguouity = pair._2.ambiguousCount * pair._3.ambiguousCount
            ambiguouity <= maxAmbiguity
          }).map(pair => {
            (pair._1, pair._2, pair._3)
          })

          nonAmbiguous
        })
    })
  }


  def toNonAmbiguousPairs(maxAmbiguity: Int, sliceSize: Int, func: (Int) => String, orders: Array[Int]): Array[(String, String, String)] = {

    orders.flatMap(order => {
      words.sliding(order, 1).toArray
        .filter(elem => elem.head.notAmbiguous())
        .flatMap(elem => {
          val aLemma = elem.head.toLemmaLabels()
          val bLemma = elem.last.toLemmaLabels()
          val aResult = aLemma ++ elem.head.toNonStemLabels(sliceSize)
          val bResult = bLemma ++ elem.last.toNonStemLabels(sliceSize)
          val linkName = "Forward-" + func(order)
          val array = if (linkName.contains("Local")) {
            aResult.zipWithIndex.flatMap { case (aLabel, aIndex) => {
              bResult.zipWithIndex.flatMap { case (bLabel, bIndex) => {
                if (bIndex > aIndex) Some((linkName, aLabel, bLabel))
                else None
              }
              }
            }
            }
          }
          else {
            aResult.flatMap(aLabel => {
              bResult.map(bLabel => {
                (linkName, aLabel, bLabel)
              })
            })
          }

          val nonAmbiguous = array.filter(pair => {
            val ambiguouity = pair._2.ambiguousCount * pair._3.ambiguousCount
            ambiguouity <= maxAmbiguity
          }).map(pair => {
            (pair._1, pair._2.tags, pair._3.tags)
          })

          nonAmbiguous
        })
    })
  }

  def toReversePairs(maxAmbiguity: Int, sliceSize: Int, func: (Int) => String, orders: Array[Int]): Array[(String, Label, Label)] = {

    orders.flatMap(order => {
      words.reverse.sliding(order, 1).toArray
        .flatMap(elem => {

          val aResult = elem.head.toNonStemLabels(sliceSize).reverse.zipWithIndex
          val bResult = elem.last.toNonStemLabels(sliceSize).reverse.zipWithIndex
          val linkName = "Backward-" + func(order)

          val array = if (linkName.contains("Local")) {
            aResult.flatMap { case (aLabel, aIndex) => {
              bResult.flatMap { case (bLabel, bIndex) => {
                if (bIndex > aIndex) Some((linkName, aLabel, bLabel))
                else None
              }
              }
            }
            }
          }
          else {
            aResult.flatMap(aLabel => {
              bResult.map(bLabel => {
                (linkName, aLabel._1, bLabel._1)
              })
            })
          }

          val nonAmbiguous = array.filter(pair => {
            val ambiguouity = pair._2.ambiguousCount * pair._3.ambiguousCount
            ambiguouity <= maxAmbiguity
          }).map(pair => {
            (pair._1, pair._2, pair._3)
          })

          nonAmbiguous
        })
    })
  }

  def toNonAmbiguosReversePairs(maxAmbiguity: Int, sliceSize: Int, func: (Int) => String, orders: Array[Int]): Array[(String, String, String)] = {

    orders.flatMap(order => {
      words.reverse.sliding(order, 1).toArray
        .filter(elem => elem.head.notAmbiguous())
        .flatMap(elem => {

          val aResult = elem.head.toNonStemLabels(sliceSize).reverse.zipWithIndex
          val bResult = elem.last.toNonStemLabels(sliceSize).reverse.zipWithIndex
          val linkName = "Backward-" + func(order)

          val array = if (linkName.contains("Local")) {
            aResult.flatMap { case (aLabel, aIndex) => {
              bResult.flatMap { case (bLabel, bIndex) => {
                if (bIndex > aIndex) Some((linkName, aLabel, bLabel))
                else None
              }
              }
            }
            }
          }
          else {
            aResult.flatMap(aLabel => {
              bResult.map(bLabel => {
                (linkName, aLabel._1, bLabel._1)
              })
            })
          }

          val nonAmbiguous = array.filter(pair => {
            val ambiguouity = pair._2.ambiguousCount * pair._3.ambiguousCount
            ambiguouity <= maxAmbiguity
          }).map(pair => {
            (pair._1, pair._2.tags, pair._3.tags)
          })

          nonAmbiguous
        })
    })
  }

}
