import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from amodal.data import make_shape_source,occlude,seed_for,source_split,save_sample,write_manifest,CompletionDataset
from amodal.model import CompletionUNet,completion_loss,sample_iou

class CoreTests(unittest.TestCase):
    def test_occlusion_targets_and_determinism(self):
        image,full=make_shape_source(43,64)
        one=occlude(image,full,123)
        two=occlude(image,full,123)
        for left,right in zip(one[:3],two[:3]):np.testing.assert_array_equal(left,right)
        result,visible,occluder,fraction=one
        self.assertTrue(.15<=fraction<=.60)
        np.testing.assert_array_equal(visible,full&~occluder)
        np.testing.assert_array_equal(result[~occluder],image[~occluder])

    def test_source_split_and_dataset_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=[]
            for i in range(60):
                sid=f'shape-{i:06d}'
                image,full=make_shape_source(seed_for(sid),64)
                for variant in range(2):records.append(save_sample(root,sid,variant,source_split(sid),image,full,42))
            write_manifest(root,records,'test')
            groups=[{r['source_id'] for r in records if r['split']==split} for split in ['train','val','test']]
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
            x,y,_=CompletionDataset(root,'train')[0]
            self.assertEqual(x.shape,(4,64,64))
            self.assertEqual(y.shape,(1,64,64))
            self.assertTrue(torch.all(x[3:4]<=y))

    def test_unet_odd_dimensions_and_gradients(self):
        model=CompletionUNet(4)
        x=torch.rand(2,4,65,71)
        y=(torch.rand(2,1,65,71)>.5).float()
        logits=model(x)
        self.assertEqual(logits.shape,y.shape)
        loss=completion_loss(logits,y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.output.weight.grad)

    def test_metrics_known_cases(self):
        full=torch.tensor([[[[True,True],[False,False]]]])
        partial=torch.tensor([[[[True,False],[False,False]]]])
        self.assertAlmostEqual(sample_iou(partial,full).item(),.5)
        self.assertEqual(sample_iou(full,full).item(),1.)
        self.assertEqual(sample_iou(full&False,full&False).item(),1.)

if __name__=='__main__':unittest.main()
