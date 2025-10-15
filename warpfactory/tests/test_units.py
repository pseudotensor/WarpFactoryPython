"""
Unit tests for warpfactory.units module

Tests physical constants and unit conversions.
"""

import pytest
from warpfactory.units import constants, length, mass, time


class TestConstants:
    """Test physical constants"""

    def test_speed_of_light(self):
        """Test speed of light constant"""
        c = constants.c()
        assert c == 2.99792458e8, "Speed of light should be 2.99792458e8 m/s"
        assert isinstance(c, float), "Speed of light should be a float"

    def test_gravitational_constant(self):
        """Test gravitational constant"""
        G = constants.G()
        assert G == 6.67430e-11, "Gravitational constant should be 6.67430e-11 m^3/kg/s^2"
        assert isinstance(G, float), "Gravitational constant should be a float"

    def test_constants_are_positive(self):
        """Verify all constants are positive"""
        assert constants.c() > 0, "Speed of light must be positive"
        assert constants.G() > 0, "Gravitational constant must be positive"

    def test_constants_immutability(self):
        """Constants should return the same value each time"""
        c1 = constants.c()
        c2 = constants.c()
        assert c1 == c2, "Speed of light should be consistent"

        G1 = constants.G()
        G2 = constants.G()
        assert G1 == G2, "Gravitational constant should be consistent"


class TestLengthUnits:
    """Test length unit conversions"""

    def test_millimeter(self):
        """Test millimeter conversion"""
        assert length.mm() == 1e-3, "1 mm should equal 1e-3 meters"

    def test_centimeter(self):
        """Test centimeter conversion"""
        assert length.cm() == 1e-2, "1 cm should equal 1e-2 meters"

    def test_meter(self):
        """Test meter (base unit)"""
        assert length.meter() == 1.0, "1 meter should equal 1 meter"

    def test_kilometer(self):
        """Test kilometer conversion"""
        assert length.km() == 1e3, "1 km should equal 1e3 meters"

    def test_length_relationships(self):
        """Test relationships between length units"""
        assert length.km() == 1000 * length.meter(), "1 km should equal 1000 m"
        assert length.meter() == 100 * length.cm(), "1 m should equal 100 cm"
        assert length.cm() == 10 * length.mm(), "1 cm should equal 10 mm"

    def test_length_conversions(self):
        """Test practical length conversions"""
        # 5 kilometers in meters
        distance_km = 5 * length.km()
        assert distance_km == 5000, "5 km should equal 5000 meters"

        # 250 centimeters in meters
        distance_cm = 250 * length.cm()
        assert distance_cm == 2.5, "250 cm should equal 2.5 meters"


class TestMassUnits:
    """Test mass unit conversions"""

    def test_gram(self):
        """Test gram conversion"""
        assert mass.gram() == 1e-3, "1 gram should equal 1e-3 kg"

    def test_kilogram(self):
        """Test kilogram (base unit)"""
        assert mass.kg() == 1.0, "1 kg should equal 1 kg"

    def test_tonne(self):
        """Test tonne conversion"""
        assert mass.tonne() == 1e3, "1 tonne should equal 1e3 kg"

    def test_mass_relationships(self):
        """Test relationships between mass units"""
        assert mass.tonne() == 1000 * mass.kg(), "1 tonne should equal 1000 kg"
        assert mass.kg() == 1000 * mass.gram(), "1 kg should equal 1000 grams"

    def test_mass_conversions(self):
        """Test practical mass conversions"""
        # 2 tonnes in kilograms
        weight_tonnes = 2 * mass.tonne()
        assert weight_tonnes == 2000, "2 tonnes should equal 2000 kg"

        # 500 grams in kilograms
        weight_grams = 500 * mass.gram()
        assert weight_grams == 0.5, "500 grams should equal 0.5 kg"


class TestTimeUnits:
    """Test time unit conversions"""

    def test_millisecond(self):
        """Test millisecond conversion"""
        assert time.ms() == 1e-3, "1 ms should equal 1e-3 seconds"

    def test_second(self):
        """Test second (base unit)"""
        assert time.second() == 1.0, "1 second should equal 1 second"

    def test_time_relationships(self):
        """Test relationships between time units"""
        assert time.second() == 1000 * time.ms(), "1 second should equal 1000 ms"

    def test_time_conversions(self):
        """Test practical time conversions"""
        # 250 milliseconds in seconds
        duration_ms = 250 * time.ms()
        assert duration_ms == 0.25, "250 ms should equal 0.25 seconds"


class TestCrossDimensionalCalculations:
    """Test calculations involving multiple unit types"""

    def test_velocity_calculation(self):
        """Test velocity = distance / time"""
        distance = 100 * length.meter()
        duration = 10 * time.second()
        velocity = distance / duration
        assert velocity == 10.0, "Velocity should be 10 m/s"

    def test_speed_of_light_with_units(self):
        """Test using speed of light with unit conversions"""
        c_m_per_s = constants.c()
        c_km_per_s = c_m_per_s / length.km()
        assert abs(c_km_per_s - 299792.458) < 1e-6, "Speed of light should be ~299792.458 km/s"

    def test_mass_energy_equivalence(self):
        """Test E = mc^2 calculation with units"""
        m = 1 * mass.kg()
        c = constants.c()
        energy = m * c * c
        expected = 8.98755178736e16  # Joules
        assert abs(energy - expected) < 1e6, "E=mc^2 calculation should be accurate"

    def test_schwarzschild_radius(self):
        """Test Schwarzschild radius calculation: rs = 2GM/c^2"""
        # For 1 solar mass (1.989e30 kg)
        M = 1.989e30 * mass.kg()
        G = constants.G()
        c = constants.c()

        rs = 2 * G * M / (c * c)
        expected_rs = 2953  # meters (approximately)

        # Should be within 1% of expected value
        relative_error = abs(rs - expected_rs) / expected_rs
        assert relative_error < 0.01, "Schwarzschild radius calculation should be accurate"

    def test_dimensional_consistency(self):
        """Verify dimensional consistency in calculations"""
        # Distance / time should give velocity
        distance = length.km()
        time_val = time.second()
        velocity = distance / time_val
        assert velocity == 1000, "km/s should equal 1000 m/s"

        # Force = mass * acceleration
        m = mass.kg()
        a = length.meter() / (time.second() ** 2)
        force = m * a
        assert force == 1.0, "1 kg * 1 m/s^2 should equal 1 Newton"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_zero_multiplication(self):
        """Test that zero times any unit is zero"""
        assert 0 * length.km() == 0
        assert 0 * mass.tonne() == 0
        assert 0 * time.ms() == 0

    def test_unit_addition(self):
        """Test addition of values in different units"""
        total_length = 1 * length.km() + 500 * length.meter() + 25 * length.cm()
        assert total_length == 1500.25, "Unit addition should work correctly"

        total_mass = 1 * mass.tonne() + 50 * mass.kg() + 250 * mass.gram()
        assert total_mass == 1050.25, "Mass addition should work correctly"

    def test_very_small_values(self):
        """Test with very small values"""
        tiny_length = 1e-10 * length.mm()
        assert tiny_length > 0, "Very small lengths should be positive"
        assert tiny_length == 1e-13, "Very small length calculation should be accurate"

    def test_very_large_values(self):
        """Test with very large values"""
        huge_distance = 1e10 * length.km()
        assert huge_distance == 1e13, "Very large distance calculation should be accurate"
