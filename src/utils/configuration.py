
# Epo 1, batch 64, pos 300. TAcc: 0.50, TRec: 0.85
# Epo 2, batch 64, pos 200. TAcc: 0.9988, TRec: 0.4865
# Epo 15, batch 64, pos 10, lr 0.001. TAcc: 0.9979, TRec: 0.8459
# Epo 55, batch 64, pos 10, lr 0.0001. TAcc: 0.9976, TRec: 0.8498

# Epo 106, batch 64, pos 10, lr 0.0001, seed 0. TAcc: 0.9962, TRec: 0.9981
# Epo 288, batch 64, pos 10, lr 0.0001, seed 2. TAcc: 0.9982, TRec: 0.9702
# Epo 111, batch 64, pos 10, lr 0.0001, seed 3. TAcc: 0.9979, TRec: 0.9995
# Epo 160, batch 64, pos 10, lr 0.0001, seed 4. TAcc: 0.9982, TRec: 0.9909
# Epo 162, batch 64, pos 10, lr 0.0001, seed 5. TAcc: 0.9988, TRec: 0.9701

# 20230710 Epo 164, batch 64, pos 10, lr 0.0001, seed 0. TAcc: 0.9965, TRec: 0.9972

elec_mlp_config = {
    'embedding': {
        'hidden_counts': [128, 128],
        'output_count': 128,
        'should_normalize': True,
        'chuanlian': {
            'channel_count': 11,
            'input_count': 6,
        },
        'rongkang': {
            'channel_count': 3257,
            'input_count': 6,
        },
        'bianya': {
            'channel_count': 3279,
            'input_count': 2,
        },
        'xiandian': {
            'channel_count': 7661,
            'input_count': 8,
        },
        'jiaoxian': {
            'channel_count': 3830,
            'input_count': 2,
        },
        'fuhe': {
            'channel_count': 6044,
            'input_count': 5,
        },
        'fadian': {
            'channel_count': 1935,
            'input_count': 8,
        },
        'muxian': {
            'channel_count': 5870,
            'input_count': 6,
        },
        'changzhan': {
            'channel_count': 1684,
            'input_count': 1,
        },
    },
    'status': {
        'input_count': 128 * 9,
        'hidden_counts': [512, 256],
        'channel_count': None,
        'output_count': 128,
        'should_normalize': True,
    },
    'classification': {
        'input_count': 128 + 128,
        'hidden_counts': [256, 128],
        'channel_count': 3619,
        'output_count': 2,
        'should_normalize': True,
    },
}


# Epo 1, batch 32. TAcc: 0.98, TRec: 0.16
# elec_mlp_config = {
#     'embedding': {
#         'hidden_counts': [256, 256],
#         'output_count': 256,
#         'should_normalize': True,
#         'chuanlian': {
#             'channel_count': 11,
#             'input_count': 6,
#         },
#         'rongkang': {
#             'channel_count': 3257,
#             'input_count': 6,
#         },
#         'bianya': {
#             'channel_count': 3279,
#             'input_count': 2,
#         },
#         'xiandian': {
#             'channel_count': 7661,
#             'input_count': 8,
#         },
#         'jiaoxian': {
#             'channel_count': 3830,
#             'input_count': 2,
#         },
#         'fuhe': {
#             'channel_count': 6044,
#             'input_count': 5,
#         },
#         'fadian': {
#             'channel_count': 1935,
#             'input_count': 8,
#         },
#         'muxian': {
#             'channel_count': 5870,
#             'input_count': 6,
#         },
#         'changzhan': {
#             'channel_count': 1684,
#             'input_count': 1,
#         },
#     },
#     'status': {
#         'input_count': 256 * 9,
#         'hidden_counts': [512, 256],
#         'channel_count': None,
#         'output_count': 256,
#         'should_normalize': True,
#     },
#     'classification': {
#         'input_count': 256 + 256,
#         'hidden_counts': [512, 256],
#         'channel_count': 3619,
#         'output_count': 2,
#         'should_normalize': True,
#     },
# }


# elec_mlp_config = {
#     'embedding': {
#         'hidden_counts': [128, 128],
#         'output_count': 128,
#         'should_normalize': True,
#         'chuanlian': {
#             'channel_count': 11,
#             'input_count': 6,
#         },
#         'rongkang': {
#             'channel_count': 3257,
#             'input_count': 6,
#         },
#         'bianya': {
#             'channel_count': 3279,
#             'input_count': 2,
#         },
#         'xiandian': {
#             'channel_count': 7661,
#             'input_count': 8,
#         },
#         'jiaoxian': {
#             'channel_count': 3830,
#             'input_count': 2,
#         },
#         'fuhe': {
#             'channel_count': 6044,
#             'input_count': 5,
#         },
#         'fadian': {
#             'channel_count': 1935,
#             'input_count': 8,
#         },
#         'muxian': {
#             'channel_count': 5870,
#             'input_count': 6,
#         },
#         'changzhan': {
#             'channel_count': 1684,
#             'input_count': 1,
#         },
#     },
#     'status': {
#         'input_count': 128 * 9,
#         'hidden_counts': [512, 256],
#         'channel_count': None,
#         'output_count': 128,
#         'should_normalize': True,
#     },
#     'classification': {
#         'input_count': 128 + 128,
#         'hidden_counts': [256, 128],
#         'channel_count': 3619,
#         'output_count': 2,
#         'should_normalize': True,
#     },
# }