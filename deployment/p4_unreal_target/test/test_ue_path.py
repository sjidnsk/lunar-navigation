import math, struct, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'joint'))
from ue_path import decode_request, encode_response, Coordinates, ReferenceGate

def test_request_centimetres_and_nonfinite():
    assert decode_request(struct.pack('<I3f',17,100,-200,300))==(17,(100.,-200.,300.))
    for data in (b'',b'0'*15,struct.pack('<I3f',17,float('nan'),0,0)):
        with pytest.raises(ValueError): decode_request(data)

def test_response_layout_and_failure():
    data=encode_response(17,0,[(100,200,300),(400,500,600)])
    assert len(data)==36
    assert struct.unpack('<IB3xI6f',data)==(17,0,2,100,200,300,400,500,600)
    assert struct.unpack('<IB3xI',encode_response(18,1,[]))==(18,1,0)
    for status,points in [(0,[]),(1,[(0,0,0)]),(3,[]),(0,[(math.nan,0,0)])]:
        with pytest.raises(ValueError):encode_response(1,status,points)

def test_coordinate_roundtrip_with_map_rotation_and_nonzero_offsets():
    c=Coordinates(.6,(21.,-35.,2.),(3.,5.,7.),(0.,0.,math.sin(.2),math.cos(.2)))
    ue=(62400.,-34400.,120.)
    assert c.to_ue(c.to_map(ue))==pytest.approx(ue,abs=1e-8)
    identity=Coordinates(0,(0,0,0),(0,0,0),(0,0,0,1))
    assert identity.to_map((100,200,300))==pytest.approx((1,-2,3))
    with pytest.raises(ValueError):Coordinates(0,(0,0,0),(0,0,0),(0,0,0,0))

def test_reference_gate_old_session_revision_invalidation_and_reactivation():
    g=ReferenceGate('a')
    assert not g.accept('b',1,0)
    assert g.accept('a',2,0)
    assert not g.accept('a',1,0)
    assert not g.accept('a',2,0)
    assert g.accept('a',2,1)
    assert not g.accept('a',2,0)
    assert g.accept('a',3,0)

def test_height_grid_column_major_and_circular_buffer():
    from ue_path import height_at
    from types import SimpleNamespace as N
    # logical increasing x/y grid [[1,2,3],[4,5,6]], width3 height2.
    m=N(header=N(frame_id='map'), layers=['elevation'],outer_start_index=1,inner_start_index=1,
        info=N(resolution=1.,length_x=3.,length_y=2.,pose=N(position=N(x=1.5,y=1.,z=0),orientation=N(x=0.,y=0.,z=0.,w=1.))))
    arr=[0.]*6
    for y in range(2):
        for x in range(3):arr[((2-1-y+1)%2)*3+(3-1-x+1)%3]=1+x+3*y
    m.data=[N(data=arr,layout=N(data_offset=0,dim=[N(label='column_index',size=2,stride=6),N(label='row_index',size=3,stride=3)]))]
    assert height_at(m,.5,.5)==1
    assert height_at(m,2.5,1.5)==6
    assert height_at(m,-.5,.5) is None
