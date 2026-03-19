package morphology.ranking

import morphology.data.Params
import sourcecode.Text.generate
import torch.{Device, Float32, Tensor}
import torch.nn.modules.Module
import torch.optim.SGD

class NMFModel(val params:Params) extends Module{
  var W = torch.rand(Seq(params.vocabSize, params.hiddenSize), requiresGrad = true)
  var H = torch.rand((Seq(params.hiddenSize, params.hiddenSize)), requiresGrad = true)

  registerParameter(W, n = "W")
  registerParameter(H, n = "H")
  
  def components(adjacency: Seq[Seq[Float]], indices:Seq[Long], threshold:Float):Array[Float] = {
    val A_sample = Tensor[Float](adjacency)
    val W_sample = W(indices)
    val W_exp1 = W_sample.unsqueeze(0)
    val W_exp2 = W_sample.unsqueeze(1)
    val A_k_all = W_exp1 * W_exp2
    val A_sample_exp = A_sample.unsqueeze(2)
    val scores = (A_k_all * A_sample_exp).sum(dim = Seq(0, 1)).div(A_sample.sum())
    val valid_k = torch.nonzero(scores > threshold).squeeze.toArray

    // Compute contribution per index
    indices.map { idx =>
      val total = valid_k.map { k =>
        W(idx, k).item
      }.sum
      total.toFloat
    }.toArray
    
  }
  
  
  def forward(A: Tensor[Float32]): Tensor[Float32]= {
    val diff = A.sub(W.matmul(H).matmul(W.t))
    val loss = (diff * diff).sum
    loss
  }
}
