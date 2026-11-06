import sys

from mlsysbook_forge.analytical import execute_analytical, executable_cells


def test_only_executable_quarto_fences_run_with_original_line_numbers():
    source = 'prose\n```python\nnot_executable()\n```\n```{python}\nx = 1\n```\n```{python}\n#| eval: false\nraise RuntimeError\n```\n'
    assert executable_cells(source) == [(6, 'x = 1\n')]


def test_observer_preserves_actual_solver_result_and_units(tmp_path):
    import mlsysim
    from mlsysbook_forge.evidence import sanitize
    kwargs = dict(model=mlsysim.Models.Vision.ResNet50, hardware=mlsysim.Hardware.Cloud.A100,
                  batch_size=1, precision='fp16')
    expected = sanitize(mlsysim.Engine.solve(**kwargs))
    path = tmp_path / 'example.py'
    path.write_text('import mlsysim\ndef main():\n    p = mlsysim.Engine.solve(model=mlsysim.Models.Vision.ResNet50, hardware=mlsysim.Hardware.Cloud.A100, batch_size=1, precision="fp16")\nmain()\n')
    previous = sys.getprofile()
    report = execute_analytical(path)
    assert sys.getprofile() is previous
    assert report['status'] == 'passed'
    assert any(row['result'] == expected for row in report['observations'])
    # MLSys·im's public Pydantic serializer retains quantities as unit strings.
    assert mlsysim.Q_(expected['latency']).check('[time]')
    assert any(row['inputs'].get('batch_size') == 1 for row in report['observations'])


def test_swallowed_external_failure_is_not_success(tmp_path):
    path = tmp_path / 'example.py'
    path.write_text('import mlsysim\ntry:\n    raise ConnectionError()\nexcept Exception:\n    pass\n')
    assert execute_analytical(path)['status'] == 'no_solver_results'
