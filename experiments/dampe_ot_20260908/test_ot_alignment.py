import io
import unittest
import numpy as np
import torch
from ot_alignment import compute_cost_matrix, sinkhorn_transport, fit_alignment, OTAlignment


class OTTests(unittest.TestCase):
    def test_rmse_against_explicit_definition(self):
        x=np.array([[1,2,5],[3,4,6]],float);y=np.array([[2,1],[2,0]],float)
        c=compute_cost_matrix(x,y)
        expected=np.array([[np.sqrt(np.mean((x[:,i]-y[:,j])**2)) for j in range(2)] for i in range(3)])
        np.testing.assert_allclose(c,expected,atol=1e-14)

    def test_analytic_two_by_two(self):
        eps=0.2
        t,meta=sinkhorn_transport([[0,1],[1,0]],epsilon=eps)
        diagonal=0.5/(1+np.exp(-1/eps))
        np.testing.assert_allclose(t,[[diagonal,0.5-diagonal],[0.5-diagonal,diagonal]],atol=1e-12)
        self.assertTrue(meta['converged'])

    def test_rectangular_and_no_information(self):
        t,_=sinkhorn_transport(np.full((3,5),1000.0))
        np.testing.assert_allclose(t,np.full((3,5),1/15),atol=1e-12)
        x=np.ones((4,3))
        np.testing.assert_allclose(x@(t*5),np.ones((4,5)),atol=1e-12)

    def test_additive_potential_invariance(self):
        c=np.array([[.2,.6,.9],[.4,.1,.8]])
        t,_=sinkhorn_transport(c,epsilon=.1,tolerance=1e-12)
        shifted,_=sinkhorn_transport(c+np.array([[100],[200]])+np.array([[10,20,30]]),epsilon=.1,tolerance=1e-12)
        np.testing.assert_allclose(t,shifted,atol=1e-10)

    def test_reject_heldout_duplicate_or_mismatched_ids(self):
        x=np.arange(6).reshape(3,2)
        for source,target,allowed in [(['a','b','test'],['a','b','test'],['a','b','c']),
                                      (['a','a','c'],['a','a','c'],['a','b','c']),
                                      (['a','b','c'],['b','a','c'],['a','b','c'])]:
            with self.assertRaises(ValueError):fit_alignment(x,x,source,target,allowed)

    def test_explicit_nonconvergence_failure(self):
        with self.assertRaises(RuntimeError):
            sinkhorn_transport([[0,.1],[0,10]],epsilon=.01,max_iter=1)

    def test_scaling_gradient_and_checkpoint(self):
        t,_=sinkhorn_transport([[0,1],[1,0]],epsilon=.05)
        layer=OTAlignment(t*2)
        x=torch.tensor([[2.,3.]],requires_grad=True)
        torch.testing.assert_close(layer(x),x@torch.tensor(t,dtype=torch.float32)*2)
        layer(x).sum().backward()
        torch.testing.assert_close(x.grad,torch.ones_like(x))
        self.assertEqual(list(layer.parameters()),[])
        buff=io.BytesIO();torch.save(layer.state_dict(),buff);buff.seek(0)
        restored=OTAlignment(torch.zeros(2,2));restored.load_state_dict(torch.load(buff,weights_only=True))
        torch.testing.assert_close(restored(x),layer(x))

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
