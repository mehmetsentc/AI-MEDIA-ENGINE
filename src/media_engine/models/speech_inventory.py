"""Pinned speech-model file inventory.

Weight checksums are the Git LFS object ids published for these revisions.
Those ids are SHA-256. The weight bytes were not downloaded.
Smaller files were hashed from their content.
"""
from __future__ import annotations

TARGETS: tuple[dict, ...] = (
    {
        'model_id': 'Systran/faster-whisper-large-v3',
        'revision': 'edaa852ec7e145841d8ffdb056a99866b5f0a478',
        'capability': 'audio.transcribe',
        'hub_used_storage': 3145762253,
        'files': (
            {'path': '.gitattributes', 'bytes': 1519, 'sha256': '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361', 'sha_source': 'content'},
            {'path': 'README.md', 'bytes': 2052, 'sha256': '39e96252229f5a3d0141dc81afb65a36fd205461ac21e5b70f2cd1248ef0082c', 'sha_source': 'content'},
            {'path': 'config.json', 'bytes': 2394, 'sha256': 'a9306624f5ec14270a014b647e5c316b6e03a662c369758d1b90697a7b0655b9', 'sha_source': 'content'},
            {'path': 'model.bin', 'bytes': 3087284237, 'sha256': '69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1', 'sha_source': 'lfs'},
            {'path': 'preprocessor_config.json', 'bytes': 340, 'sha256': '7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711', 'sha_source': 'content'},
            {'path': 'tokenizer.json', 'bytes': 2480617, 'sha256': '6d8cbd7cd0d8d5815e478dac67b85a26bbe77c1f5e0c6d76d1ce2abc0e5f21ca', 'sha_source': 'content'},
            {'path': 'vocabulary.json', 'bytes': 1068114, 'sha256': 'c69260f2ab26d659b7c398f9a2b2b48ed0df16c3b47d7326782fd9cba71690c1', 'sha_source': 'content'},
        ),
    },
    {
        'model_id': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        'revision': '0c0e3051f131929182e2c023b9537f8b1c68adfe',
        'capability': 'audio.tts',
        'hub_used_storage': 4617449338,
        'files': (
            {'path': '.gitattributes', 'bytes': 1519, 'sha256': '11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361', 'sha_source': 'content'},
            {'path': 'README.md', 'bytes': 57846, 'sha256': '64f65e809b51cc0c35f393fbbfcc2d735c0cb3fdbbc6f3fdc4a5e6ce55e9d088', 'sha_source': 'content'},
            {'path': 'config.json', 'bytes': 4908, 'sha256': '17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9', 'sha_source': 'content'},
            {'path': 'generation_config.json', 'bytes': 245, 'sha256': 'f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa', 'sha_source': 'content'},
            {'path': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3', 'sha_source': 'content'},
            {'path': 'model.safetensors', 'bytes': 3833402552, 'sha256': '38b1d5971bdbd982b561cccec982669a53b0537c3cf5e9bd4778ed07bb2f5137', 'sha_source': 'lfs'},
            {'path': 'preprocessor_config.json', 'bytes': 127, 'sha256': 'efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/config.json', 'bytes': 2336, 'sha256': 'ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/configuration.json', 'bytes': 76, 'sha256': '6bc26d64eb5024b4d1dab5a52371958b429256d6c9d59787f1f5294a54e0cebd', 'sha_source': 'content'},
            {'path': 'speech_tokenizer/model.safetensors', 'bytes': 682293092, 'sha256': '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258', 'sha_source': 'lfs'},
            {'path': 'speech_tokenizer/preprocessor_config.json', 'bytes': 234, 'sha256': 'fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb', 'sha_source': 'content'},
            {'path': 'tokenizer_config.json', 'bytes': 7344, 'sha256': 'dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670', 'sha_source': 'content'},
            {'path': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 'sha_source': 'content'},
        ),
    },
)

