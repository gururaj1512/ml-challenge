import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from erfast.common import load_sparse,save_sparse
from erfast.lexical import make_postings
from erfast.wand import search
from erfast.features import FEATURES,make_features
from erfast.metrics import entity_f05,choose_budget
from erfast.pairs import fuse
from erfast.text import normalize_text,accent_fold,structured_tokens,record_from_row

class Core(unittest.TestCase):
    def test_unicode(self):
        self.assertEqual(normalize_text('Cozy [Gríll]'),'cozy gríll')
        self.assertIn('म',normalize_text('मॉडर्न'))
        self.assertEqual(accent_fold('Gríll'),'grill')
    def test_structured(self):
        self.assertEqual(structured_tokens('A702 G-1 14 /15 5A1'),{'a702','g-1','14/15','5a1'})
    def test_empty(self):
        self.assertEqual(normalize_text(None),'')
    def test_metric(self):
        self.assertEqual(entity_f05([],[]),1)
        self.assertEqual(entity_f05([],['x']),0)
        self.assertAlmostEqual(entity_f05(['a','b'],['a','b','c']),5/7)
    def test_fusion(self):
        kept,ranks,raw=fuse({'name':[1,2,3],'embedding':[2,4]},60,2)
        self.assertEqual(kept[0],2);self.assertEqual(len(raw),4);self.assertIn('embedding',ranks[2])
    def test_budget(self):
        r=[{'candidate_budget':16,'mean_candidates':16,'macro_f05':.94,'candidate_link_recall':.95},
           {'candidate_budget':48,'mean_candidates':44,'macro_f05':.9405,'candidate_link_recall':.951}]
        self.assertEqual(choose_budget(r,.001,.002)['candidate_budget'],16)
    def test_wand_random(self):
        rng=np.random.default_rng(9)
        for trial in range(40):
            t=sparse.random(250,90,density=.05+.01*(trial%10),format='csr',random_state=rng,dtype=np.float32)
            q=sparse.random(30,90,density=.12,format='csr',random_state=rng,dtype=np.float32)
            if trial%2==0:t=normalize(t);q=normalize(q)
            p,m=make_postings(t);ids,score=search(p,m,q,10,2);ref=(q@t.T).toarray()
            for i in range(len(ids)):
                ix=np.argsort(-ref[i],kind='stable')[:10];v=ref[i,ix];valid=v>0
                np.testing.assert_allclose(score[i,:sum(valid)],v[valid],rtol=1e-5,atol=2e-6)
                self.assertEqual(len(set(ids[i][ids[i]>=0])),sum(ids[i]>=0))
    def test_wand_ties(self):
        t=sparse.csr_matrix(np.ones((12,3),np.float32));q=t[:1];p,m=make_postings(t)
        ids,_=search(p,m,q,4,1);self.assertEqual(ids.tolist(),[[0,1,2,3]])
    def test_wand_zero(self):
        t=sparse.csr_matrix(np.ones((12,3),np.float32));q=sparse.csr_matrix((2,3),dtype=np.float32);p,m=make_postings(t)
        ids,s=search(p,m,q,4,1);self.assertTrue(np.all(ids==-1));self.assertTrue(np.all(s==0))
    def test_wand_lexical(self):
        v=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),dtype=np.float32)
        t=v.fit_transform(['Caldeon Nova LLC','Caldeon Nova','Different business','Cozy Grill','Cozy Gríll'])
        q=v.transform(['Caldeon Nova','Cozy Grill']);p,m=make_postings(t);ids,s=search(p,m,q,3,1)
        ref=(q@t.T).toarray()
        for i in range(2):np.testing.assert_allclose(s[i],np.sort(ref[i])[::-1][:3],atol=1e-6)
    def test_sparse_roundtrip(self):
        x=sparse.csr_matrix(np.array([[1,0,2],[0,0,3]],dtype=np.float32))
        with tempfile.TemporaryDirectory() as d:
            save_sparse(d,x);y=load_sparse(d);np.testing.assert_array_equal(x.toarray(),y.toarray())
    def test_schema_features(self):
        class Lex:
            def cosine(self,a,b,f):return .8
            def name_frequency(self,n):return 2
        a=record_from_row({'entity_id':'S1-1','business_name':'ABC Pvt Ltd','business_address':'G-1 Pune','country':'India'})
        b=record_from_row({'entity_id':'S2-1','business_name':'ABC Private Limited','business_address':'G-1 Pune','country':'India'})
        d=make_features(a,b,{'name':1,'embedding':2},Lex(),.9)
        self.assertEqual(list(d),FEATURES);self.assertEqual(len(d),48);self.assertEqual(d['name_stripped_exact'],1.)
        self.assertTrue(np.isfinite(list(d.values())).all())

if __name__=='__main__':unittest.main()
