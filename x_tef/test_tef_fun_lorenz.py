#!/usr/bin/env python3

import unittest
import numpy as np
import tef_fun_lorenz as tfl

def val_at(a1, a2, v2):
    assert a1.ndim == 1
    assert a2.ndim == 1
    assert len(a1) == len(a2)
    i2 = (a2 == v2).nonzero()[0][0]
    return a1[i2]

class TestTefFunLorenz(unittest.TestCase):
    def test_bulk_calc_2lay(self):
        # Define a simple two-layer isohaline function Q(s) based on an
        # inflow at salinity b, ocean salinity c, and net flow of qr.
        qr = -1000
        b = 25
        c = 35

        # Derivation is in test_tef_fun_lorenz.md
        A = 420 * qr / (7*b*c**6 - 4*c**7)
        q0 = -c**6/60 + b*c**5/30
        Q = np.polynomial.polynomial.Polynomial(A*np.array([
            q0,
            0,
            0,
            -b*c**2/3,
            (2*b*c+c**2)/4,
            -(b+2*c)/5,
            1/6
        ]))

        s = np.linspace(0, 35, 10000)
        tef_q = Q(s)

        Q_in, Q_out, s_in, s_out, div_sal, ind, min_max = tfl.calc_bulk_values(s, tef_q, tef_q*s)

        # Check shapes of results
        self.assertEqual(len(Q_in), len(Q_out))
        self.assertEqual(len(Q_in), len(s_in))
        self.assertEqual(len(Q_in), len(s_out))
        self.assertEqual(len(div_sal), 3)

        # Check that values match what was used to derive the TF
        self.assertAlmostEqual(s_in[0], b, places=2)
        self.assertAlmostEqual(div_sal[1], b, places=2)
        self.assertAlmostEqual(div_sal[-1], c, places=2)
        self.assertAlmostEqual(Q_in[0], Q(b), places=2)

    def test_find_extrema_trivial(self):
        """Trivial case"""

        qv = np.array([7802, 7802, 7802, 7838, 7865, 7865, 7865])

        ind, minmax = tfl.find_extrema(qv)

        self.assertTrue(np.all(np.diff(ind) > 0),
            "indices are not monotonically increasing")
        self.assertFalse('minmin' in ''.join(minmax),
            "consecutive minima returned")
        self.assertFalse('maxmax' in ''.join(minmax),
            "consecutive maxima returned")

        self.assertEqual(len(ind), 2)
        self.assertEqual(ind[0], 2)
        self.assertEqual(minmax[0], 'min')
        self.assertEqual(ind[1], 4)
        self.assertEqual(minmax[1], 'max')

    def test_find_extrema_easy(self):
        """Multi layer test case for find_extrema"""

        qv = np.array([  38.5,   38.5,   38.5,   38.5,   38.5,   38.5,   38.5,
                         38.5,   38.5,   38.5,   36.6,   35.8,   32.3,   31.6,
                         31.6,   29.4,   28.5,   27.5,   26.2,   26.2,   26.2,
                         24.6,   23.4,   23.4,   23.4,   23.4,   22.7,   22.1,
                         19.3,   17.8,   17.8,   17.8,   17.3,   15.9,   15.9,
                         14. ,   14. ,   14. ,   14. ,   12.9,   12.9,   12.9,
                         12.9,   12.9,   12.9,   12.3,   12.3,   10.3,    9.1,
                          8.5,    7. ,    7. ,    7. ,    7. ,    7. ,    5.7,
                          5.7,    5. ,    5. ,    2.5,    2. ,    2. ,    2. ,
                          0.5,    0.5,    0.5,    0.5,   -1.3,   -1.3,   -2.5,
                         -3.2,   -4.8,   -8.6,  -10.1,  -10.1,  -10.1,  -10.1,
                        -14.1,  -15.6,  -17.2,  -23.3,  -26.8,  -34.6,  -38.1,
                        -48.2,  -54.7,  -65.4,  -67.2,  -81.8,  -97.2, -119. ,
                       -138.2, -149.5, -186.2, -216.3, -230.3, -238. , -262.3,
                       -265. , -279.8, -289.3, -294.5, -305.8, -298.6, -298.9,
                       -295. , -300.5, -300.3, -306.2, -303.5, -301.8, -307.1,
                       -307.1, -314.4, -314.1, -314.6, -318.3, -313.9, -308.9,
                       -307.5, -295.6, -277.2, -259.6, -241.5, -215.4, -200.5,
                       -194.8, -154.1, -132.7, -115.5,  -78.7,  -69.9,  -58.6,
                        -57.8,  -56.9,  -59.7,  -60.1,  -60.7,  -61.2,  -61.2,
                        -58.7,  -59. ,  -60.6,  -58. ,  -55.2,  -42.9,  -34.8,
                        -22.6,   -7.3,    8.3,   18.7,   29.7,   49.4,   81.1,
                         90. ,  106.3,  132.3,  143.6,  151.8,  153.9,  153.8,
                        161.2,  162.8,  164. ,  164.3,  166.6,  156.9,  169.1,
                        164.6,  163.9,  155.5,  155.5,  157.1,  149. ,  153.9,
                        171.6,  178.6,  185.8,  194.8,  196.7,  202.3,  203. ,
                        206.7,  199.7,  192. ,  199.1,  180.2,  164. ,  168.9,
                        129.7,   99.1,   47.1,   52.7,   41.9,   51.8,   -5.5,
                        -28.1,  -58.2,  -87.7,  -87.5, -139.3, -207.7, -265.7,
                       -226.2, -232.9, -347.7, -189.2, -112.8,    0. ,    0. ,
                          0.])

        ind, minmax = tfl.find_extrema(qv)

        self.assertTrue(np.all(np.diff(ind) > 0),
            "indices are not monotonically increasing")
        self.assertFalse('minmin' in ''.join(minmax),
            "consecutive minima returned")
        self.assertFalse('maxmax' in ''.join(minmax),
            "consecutive maxima returned")

        self.assertIn(9, ind)
        self.assertIn(116, ind)
        self.assertIn(182, ind)
        self.assertIn(205, ind)
        self.assertIn(208, ind)

        self.assertEqual(val_at(minmax, ind, 9), 'max')
        self.assertEqual(val_at(minmax, ind, 116), 'min')
        self.assertEqual(val_at(minmax, ind, 182), 'max')
        self.assertEqual(val_at(minmax, ind, 205), 'min')
        self.assertEqual(val_at(minmax, ind, 208), 'max')

    def test_find_extrema_asym(self):
        """Complex signal that find_peaks may have trouble with"""

        qv = np.array([7802, 7802, 7802, 7802, 7802, 7802, 7838, 7865, 7865, 
            7865, 7865, 7865, 7865, 7865, 7865, 7865, 7865, 7865, 7865, 7865,
            7865, 7865, 7865, 7889, 7889, 7889, 7889, 7889, 7889, 7889, 7889,
            7889, 7889, 7889, 7889, 7889, 7889, 7889, 7904, 7904, 7904, 7904,
            7904, 7904, 7904, 7935, 7935, 7935, 7940, 7940, 7940, 7966, 7966,
            7966, 7966, 7966, 7966, 7966, 7966, 7966, 7966, 7966, 7966, 8006,
            8006, 8006, 8040, 8040, 8040, 8040, 8040, 8040, 8040, 8040, 8040,
            8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042,
            8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042, 8042,
            8042, 8042, 8042, 8042, 8042, 8042, 8042, 8024, 8024, 8024, 8036,
            8036, 8036, 8036, 8036, 8036, 8036, 8069, 8069, 8069, 8069, 8070,
            8070, 8070, 8070, 8070, 8070, 8070, 8070, 8070, 8070, 8070, 8070,
            8055, 8055, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097,
            8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097, 8097,
            8097, 8097, 8097, 8097, 8097, 8097, 8115, 8115, 8144, 8144, 8144,
            8144, 8144, 8144, 8144, 8144, 8144, 8144, 8143, 8143, 8143, 8143,
            8143, 8143, 8143, 8143, 8143, 8157, 8157, 8157, 8157, 8157, 8157,
            8156, 8156, 8156, 8156, 8156, 8156, 8156, 8156, 8156, 8156, 8177,
            8177, 8177, 8177, 8177, 8177, 8177, 8177, 8177, 8177, 8244, 8244,
            8244, 8226, 8226, 8226, 8245, 8245, 8245, 8245, 8245, 8246, 8246,
            8246, 8213, 8213, 8213, 8213, 8213, 8213, 8213, 8249, 8249, 8249,
            8249, 8249, 8249, 8249, 8249, 8249, 8249, 8249, 8241, 8241, 8241,
            8241, 8288, 8264, 8264, 8264, 8263, 8292, 8292, 8292, 8292, 8292,
            8333, 8333, 8333, 8333, 8338, 8338, 8338, 8338, 8338, 8338, 8355,
            8355, 8360, 8360, 8363, 8363, 8364, 8364, 8364, 8364, 8364, 8364,
            8364, 8364, 8364, 8364, 8363, 8363, 8439, 8439, 8439, 8451, 8507,
            8508, 8534, 8534, 8506, 8506, 8506, 8507, 8507, 8507, 8468, 8468,
            8469, 8469, 8489, 8503, 8471, 8471, 8467, 8484, 8485, 8484, 8486,
            8486, 8507, 8505, 8505, 8505, 8505, 8505, 8505, 8505, 8505, 8527,
            8527, 8526, 8526, 8526, 8526, 8553, 8574, 8622, 8622, 8622, 8620,
            8620, 8620, 8579, 8578, 8578, 8578, 8588, 8583, 8583, 8583, 8583,
            8583, 8583, 8602, 8603, 8567, 8605, 8547, 8547, 8623, 8610, 8569,
            8569, 8631, 8631, 8633, 8632, 8632, 8602, 8600, 8600, 8600, 8600,
            8600, 8622, 8622, 8623, 8649, 8649, 8647, 8647, 8631, 8643, 8620,
            8626, 8625, 8658, 8659, 8657, 8657, 8655, 8623, 8643, 8630, 8624,
            8621, 8658, 8710, 8668, 8668, 8691, 8700, 8700, 8700, 8698, 8698,
            8714, 8817, 8824, 8834, 8835, 8823, 8769, 8772, 8825, 8872, 8884,
            8884, 8884, 8954, 8952, 8914, 8913, 8926, 8919, 8953, 8955, 9033,
            9029, 8964, 8904, 8939, 8997, 9025, 9025, 8997, 8993, 9137, 9135,
            9127, 9127, 9126, 9127, 9093, 9153, 9153, 9154, 9157, 9207, 9317,
            9293, 9312, 9308, 9306, 9454, 9450, 9510, 9508, 9444, 9391, 9349,
            9391, 9440, 9382, 9434, 9455, 9545, 9566, 9707, 9683, 9693, 9708,
            9746, 9781, 9867, 9866, 9960, 10023, 10018, 9970, 9997, 9999,
            10143, 10170, 10266, 10142, 10172, 10123, 10238, 10294, 10443,
            10471, 10560, 10714, 10656, 10631, 10697, 10514, 10662, 10696,
            10712, 10763, 10873, 10873, 10799, 10698, 10587, 10747, 10872,
            10808, 11009, 11032, 11301, 11245, 11427, 11608, 11940, 11894,
            12193, 12263, 12651, 12765, 12730, 13005, 13543, 13650, 13837,
            14033, 14190, 14400, 14739, 15001, 15399, 15884, 15887, 16536,
            16524, 16524, 16524, 16722, 16761, 16658, 16586, 16766, 16979,
            17371, 17883, 18553, 18755, 19404, 19847, 20171, 19912, 19987,
            20007, 20004, 20304, 21068, 20863, 21467, 21886, 22712, 23391,
            23565, 24240, 24383, 25205, 25935, 26241, 26632, 26755, 27372,
            27869, 28244, 29223, 29551, 29958, 30445, 30598, 30523, 31136,
            31297, 32025, 32006, 32329, 33185, 33562, 34045, 35059, 35159,
            35710, 36043, 36590, 37241, 37586, 37816, 37880, 38720, 39226,
            39307, 40179, 40425, 40368, 40193, 40060, 41019, 41471, 41896,
            42641, 42367, 42685, 43320, 43849, 43765, 44193, 44921, 44883,
            45076, 44794, 45277, 46403, 44879, 44275, 43864, 44005, 44752,
            44153, 44933, 45152, 45315, 44877, 44099, 44025, 44260, 44138,
            44156, 44832, 45551, 47146, 47915, 47388, 46757, 47195, 46911,
            47325, 47094, 46247, 45921, 46406, 46959, 46868, 46658, 46978,
            45546, 44883, 45075, 45313, 44991, 44798, 45005, 44738, 43664,
            42911, 43040, 42782, 42133, 42291, 42657, 42368, 42386, 42346,
            42094, 42090, 41798, 41355, 41450, 41180, 40804, 40491, 39131,
            38570, 37879, 37542, 37443, 36843, 36782, 36283, 36265, 35960,
            35009, 34834, 34407, 33845, 33434, 32973, 33451, 32777, 31940,
            31417, 30849, 30296, 30249, 29615, 29520, 29289, 28447, 28045,
            26679, 26659, 25744, 25781, 25266, 24746, 24265, 23199, 23425,
            22256, 22183, 21627, 20692, 19312, 18526, 18092, 17360, 16839,
            15345, 15303, 14695, 14561, 13316, 12705, 11041, 10155, 9461,
            8131, 7323, 6218, 5231, 4497, 3597, 2913, 2565, 1843, 1183, 917,
            697, 604, 413, 349, 191, 124, 85, 70, 45, 28, 28, 28, 10, 0, 0,
            0, 0, 0,])

        ind, minmax = tfl.find_extrema(qv)

        self.assertTrue(np.all(np.diff(ind) > 0),
            "indices are not monotonically increasing")
        self.assertFalse('minmin' in ''.join(minmax),
            "consecutive minima returned")
        self.assertFalse('maxmax' in ''.join(minmax),
            "consecutive maxima returned")

        self.assertIn(5, ind)
        self.assertIn(611, ind)
        self.assertIn(627, ind)
        self.assertIn(746, ind)

        self.assertEqual(val_at(minmax, ind, 5), 'min')
        self.assertEqual(val_at(minmax, ind, 611), 'min')
        self.assertEqual(val_at(minmax, ind, 627), 'max')
        self.assertEqual(val_at(minmax, ind, 746), 'min')

    def test_find_extrema_hard(self):
        """Test case for known failing transport function in orig code"""

        qv = np.array([  16.3,   16.3,   16.3,   16.3,   16.3,   16.3,   16.3,
                         16.3,   16.3,   16.3,   16.2,   16.2,   16. ,   15.9,
                         15.6,   15.5,   15.2,   15.2,   14.4,   13.5,   11.4,
                         10. ,    9.1,    6.3,    3. ,    1.2,    0.5,   -2.4,
                         -3. ,   -3.8,   -4.7,   -4.4,   -5.5,   -0.2,    1. ,
                          4.9,    5.8,    7. ,    7.4,    9.3,   10.6,   10.1,
                         10.1,    8.5,    9.7,    9.8,    9.1,   11.8,   26.9,
                         56.4,   83.6,  139.4,  195.2,  233.7,  262.2,  286.4,
                        309.1,  374.7,  416.3,  457.3,  527.6,  566.9,  613.7,
                        631.5,  641.6,  650. ,  658.1,  661.5,  664.2,  663. ,
                        673.2,  674.5,  675.4,  680.7,  686.5,  703.5,  717.5,
                        731.2,  761.4,  773.5,  805.2,  817.4,  847. ,  882.6,
                        903.4,  917.2,  955.5,  963.1,  985.1,  995.8,  991.3,
                        994. ,  999.4,  997.6, 1002.1, 1009.2, 1004.6, 1008.6,
                       1009.5, 1007.9, 1007.2, 1008.2, 1010.3, 1007.2, 1010.1,
                       1017.5, 1020.9, 1020.9, 1031.7, 1037.1, 1040.1, 1033.9,
                       1047.1, 1046. , 1038.3, 1040.9, 1037.2, 1033.4, 1035.5,
                       1024.7, 1014.5, 1000.3,  983.8,  984.2,  977.9,  933.1,
                        930.8,  904.7,  868.2,  837.3,  806.7,  761.6,  655.7,
                        575. ,  523.7,  342.4,  287.7,  170.9,    0. ,    0. ])

        ind, minmax = tfl.find_extrema(qv)

        self.assertTrue(np.all(np.diff(ind) > 0),
            "indices are not monotonically increasing")
        self.assertFalse('minmin' in ''.join(minmax),
            "consecutive minima returned")
        self.assertFalse('maxmax' in ''.join(minmax),
            "consecutive maxima returned")

        self.assertIn(qv.argmax(), ind)
        self.assertIn(qv.argmin(), ind)

if __name__ == '__main__': unittest.main()
