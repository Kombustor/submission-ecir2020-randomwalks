import xml.etree.ElementTree as ET

def port_layout(source_gefx, target_gefx, save_gefx):
    source = ET.parse(source_gefx).getroot()
    target = ET.parse(target_gefx).getroot()

    print("Source", source)
