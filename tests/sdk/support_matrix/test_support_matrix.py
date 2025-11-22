from aiconfigurator.sdk.support_matrix import SupportMatrix


def test_system_and_backend_matches_database():
    """
    Test that the system and backend defined in the support matrix matches the database.
    """
    support_matrix = SupportMatrix()
    systems_in_database = set(support_matrix.databases.keys())
    backends_in_database = {backend for system in systems_in_database for backend in support_matrix.databases[system]}
    assert systems_in_database == support_matrix.get_systems()
    assert backends_in_database == support_matrix.get_backends()
