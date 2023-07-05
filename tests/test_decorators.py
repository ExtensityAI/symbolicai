import os
import unittest

from examples.demo import Demo
from symai.symbol import Symbol

# for debugging
# attention this constantly overwrites the keys config file
#os.environ['OPENAI_API_KEY'] = ''


class TestDecorator(unittest.TestCase):
    def test_few_shot(self):
        demo = Demo()
        names = demo.generate_japanese_names()
        self.assertIsNotNone(names)
        self.assertTrue(len(names) == 2)
        print(names)
        res = demo.is_name('Japanese', names)
        self.assertTrue(res)
        res = demo.is_name('German', names)
        self.assertFalse(res)

    def test_zero_shot(self):
        demo = Demo()
        val = demo.get_random_int()
        self.assertIsNotNone(val)
        print(val)

    def test_equals(self):
        demo = Demo(2)
        res = demo.equals_to('2')
        self.assertTrue(res)

    def test_compare(self):
        demo = Demo(175)
        res = demo.larger_than(66)
        self.assertTrue(res)

    def test_rank(self):
        demo = Demo(['1.66m', '1.75m', '1.80m'])
        res = demo.rank_list('hight', ['1.75m', '1.66m', '1.80m'], order='asc')
        self.assertTrue(demo.equals_to(res))

    def test_case(self):
        demo = Demo('angry')
        res = demo.sentiment_analysis('I really hate this stupid application because it does not work.')
        self.assertTrue(demo.equals_to(res))

    def test_translate(self):
        demo = Demo()
        res = demo.translate('I feel tired today.', language='Spanish')
        self.assertIsNotNone(res)

    def test_extract_pattern(self):
        demo = Demo('Open the settings.json file, edit the env property and run the application again with the following command: python main.py')
        res = demo.extract_pattern('Files with *.json')
        self.assertTrue(res == 'settings.json')

    def test_replace(self):
        demo = Demo('Steve Ballmer is the CEO of Microsoft.')
        res = demo.replace_substring('Steve Ballmer', 'Satya Nadella')
        self.assertTrue('Satya' in res)

    def test_expression(self):
        demo = Demo(18)
        res = demo.evaluate_expression('2 + 4 * 2 ^ 2')
        self.assertTrue(demo.value == res)

    def test_notify_subscriber(self):
        demo = Demo()
        res = demo.notify_subscriber('You can contact us via email at office@alphacore.eu',
                                     subscriber={'europe': lambda x: Exception('Not allowed')})
        self.assertTrue('email' in res)

    def test_try(self):
        sym = Symbol("""# generate method
def generate_mesh():
    # imports
    import bpy
    import bmesh
    import os

    # clean scene
    bpy.ops.object.select_all(action='SELECT')
    # Delete selected objects
    bpy.ops.object.delete()

    # create bmesh
    bm = bmesh.new()

    # custom code
    import bpy
    from mathutils import Matrix

    # create box house with height 8, width 10, depth 10;
    house = bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, 0))
    bpy.ops.transform.resize(value=(10, 10, 8))

    # create roof for house with height 4, width 12, depth 12, pitch 30;
    roof = bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=6, radius2=0, depth=4, enter_editmode=False, location=(0, 0, 8))
    bpy.ops.transform.rotate(value=0.523599, orient_axis='X')

    # create box door with height 6, width 3, depth 1.5;
    door = bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, 0))
    bpy.ops.transform.resize(value=(3, 1.5, 6))

    # place door on house with bottom at 0, left at 2, front at 0;
    door_location = Matrix.Translation((2, 5, 0)) @ Matrix.Rotation(1.5708, 4, 'Y')
    door.location = door_location @ house.location

    # create box window with height 4, width 4, depth 1.5;
    window = bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, location=(0, 0, 0))
    bpy.ops.transform.resize(value=(4, 1.5, 4))

    # place window on house with bottom at 2, left at 2, front at 5;
    window_location = Matrix.Translation((2, 2.75, 5)) @ Matrix.Rotation(1.5708, 4, 'Y')
    window.location = window_location @ house.location

    # create cylinder chimney with radius 0.5, height 2;
    chimney = bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2, enter_editmode=False, location=(0, 0, 0))

    # place chimney on house with top at 8, left at 5, front at 5;
    chimney_location = Matrix.Translation((5, 5, 8))
    chimney.location = chimney_location @ house.location

    # Check for collisions
    if house.collision.dop_objects_colliding:
        print("Objects are colliding!")

    # export meshes
    os.makedirs("results", exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.export_scene.obj(filepath="results/tmp.obj", use_selection=True)

    # return bmesh
    return bm

res = generate_mesh()
""")
        res = sym.fexecute()
        self.assertTrue(res['locals']['res'] is not None, res)


if __name__ == '__main__':
    unittest.main()
