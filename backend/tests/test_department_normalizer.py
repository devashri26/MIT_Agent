from backend.ingestion.normalizers.department_normalizer import DepartmentNormalizer


def test_department_normalizer_removes_faculty_noise() -> None:
    normalizer = DepartmentNormalizer()

    assert normalizer.normalize("Computer Engineering Dr") == "Computer Engineering"
    assert normalizer.normalize("Information Technology Mrs") == "Information Technology"
    assert normalizer.normalize("it") == "Information Technology"

