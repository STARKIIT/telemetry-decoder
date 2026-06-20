import os
import sys
from fastapi.testclient import TestClient

# Setup sys.path so test runner can find sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "device" in json_data
    assert "pipeline_dir" in json_data
    assert "code" in json_data

def test_run_endpoint_default():
    payload = {
        "message": "TEST MESSAGE 123",
        "interference": "none",
        "sjr_db": 10.0,
        "snr_db": 25.0,
        "seed": 42
    }
    response = client.post("/api/run", json=payload)
    assert response.status_code == 200
    res = response.json()
    
    # Assert correctness of response schema
    assert res["ok"] is True
    assert res["message_sent"] == "TEST MESSAGE 123"
    assert "recovered_pre_fec" in res
    assert "recovered_post_fec" in res
    
    assert "stats" in res
    stats = res["stats"]
    assert "n_frames" in stats
    assert "raw_ber" in stats
    assert "total_raw_errors" in stats
    assert "sync_err_samples" in stats
    assert "psl" in stats
    
    assert "stages" in res
    assert len(res["stages"]) == 4
    
    # Check stage 1 details
    stage1 = res["stages"][0]
    assert stage1["id"] == 1
    assert "name" in stage1
    assert "time" in stage1
    assert "i" in stage1["time"]
    assert "q" in stage1["time"]
    assert len(stage1["time"]["i"]) == 400
    assert "spectrum" in stage1
    assert "freq_mhz" in stage1["spectrum"]
    assert "psd_db" in stage1["spectrum"]
    assert len(stage1["spectrum"]["freq_mhz"]) == 256
    
    assert "frames" in res
    assert len(res["frames"]) > 0
    assert "index" in res["frames"][0]
    assert "raw_errors" in res["frames"][0]
    assert "sync_err" in res["frames"][0]
    assert "psl" in res["frames"][0]

def test_run_endpoint_wideband():
    payload = {
        "message": "ALTITUDE 35000 SPEED 480 HEADING 270",
        "interference": "wideband",
        "sjr_db": 0.0,
        "snr_db": 15.0,
        "seed": 42
    }
    response = client.post("/api/run", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["ok"] is True
    assert res["recovered_post_fec"] == "ALTITUDE 35000 SPEED 480 HEADING 270"
