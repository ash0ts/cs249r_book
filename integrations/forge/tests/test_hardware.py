import json

import pytest

from mlsysbook_forge.hardware import analyze


def test_dataset_manifest_detects_leakage_without_uploading_identifiers():
    good = b'sample_id,label,split\nprivate-id-1,private-class,train\nprivate-id-2,private-class,test\n'
    result = analyze(good, 'dataset', platform='Nicla Vision')
    assert result['split_counts'] == {'train':1,'test':1}
    assert 'private-' not in json.dumps(result)
    with pytest.raises(ValueError, match='leakage'):
        analyze(good.replace(b'private-id-2',b'private-id-1'), 'dataset', platform='Nicla Vision')


def test_model_storage_does_not_claim_activation_ram_or_device_execution():
    source = {'model_sha256':'a'*64,'parameter_tensors':[{'shape':[3,4],'dtype':'int8'},{'shape':[4],'dtype':'float32'}], 'flash_budget_bytes':20}
    result = analyze(json.dumps(source).encode(),'model',platform='XIAO')
    assert result['parameter_count'] == 16
    assert result['parameter_storage_bytes'] == 28
    assert result['parameter_bytes_fit_flash'] is False
    assert result['result_kind'] == 'supplied_case'


def test_imported_measurements_preserve_units_and_lineage():
    source = {'model_sha256':'a'*64,'firmware_sha256':'b'*64,'board':'Raspberry Pi 5','capture_method':'device timer after warmup',
              'measurements':[{'latency_ms':10},{'latency_ms':14}]}
    result = analyze(json.dumps(source).encode(),'measurements',platform='Raspberry Pi')
    assert result['summary']['latency_ms']['mean'] == 12
    assert result['summary']['latency_ms']['unit'] == 'ms'
    assert result['firmware_sha256'] == 'b'*64
    assert result['result_kind'] == 'measured_experiment'
    source['measurements'][0]['latency_ms'] = float('nan')
    with pytest.raises(ValueError,match='finite'):
        analyze(json.dumps(source).encode(),'measurements',platform='Raspberry Pi')
