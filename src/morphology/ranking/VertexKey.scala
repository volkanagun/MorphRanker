package morphology.ranking

case class VertexKey(var label:String, var value:String) extends Serializable
{
  override def hashCode(): Int = {
    var r= 7;
    if(value!=null && label!=null) {
      r * label.hashCode() + value.hashCode
    }
    else r
  }

  def isDistant():Boolean={
    label.startsWith("Distant")
  }
  def isLocal():Boolean={
    label.startsWith("Local")
  }
  def isNeighbour():Boolean={
    label.startsWith("Neighbour")
  }

  override def equals(obj: Any): Boolean = {
    if(obj.isInstanceOf[VertexKey]) {
      val otherKey = obj.asInstanceOf[VertexKey]
      otherKey.label.equals(label) && otherKey.value.equals(value)
    }
    else{
      false
    }
  }
}
