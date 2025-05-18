-- DROP TABLE IF EXISTS "palettes";

CREATE TABLE IF NOT EXISTS "palettes" (
  "id" INTEGER,
  "likes" INTEGER,
  "submission_date" TEXT,
  "color_hunt_id" TEXT UNIQUE ON CONFLICT REPLACE,
  PRIMARY KEY ("id")
);

-- DROP TABLE IF EXISTS "palette_tags";

CREATE TABLE IF NOT EXISTS "palette_tags" (
  "name" TEXT,
  "is_colour_tag" BOOLEAN,
  PRIMARY KEY ("name")
);

-- DROP TABLE IF EXISTS "palette_tag_associations";

CREATE TABLE IF NOT EXISTS "palette_tag_associations" (
  "palette_id" INTEGER,
  "tag" TEXT,
  PRIMARY KEY ("palette_id", "tag")
  FOREIGN KEY ("tag") REFERENCES "palette_tags" ("name")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
);

-- DROP TABLE IF EXISTS "palette_colours";

CREATE TABLE IF NOT EXISTS "palette_colours" (
  "palette_id" INTEGER,
  "colour" CHAR(7),
  PRIMARY KEY ("palette_id", "colour")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
);

--DROP TABLE IF EXISTS "clothes";

CREATE TABLE IF NOT EXISTS "clothes" (
  "source" TEXT,
  "item_id" TEXT,
  "title" TEXT,
  "image_url" TEXT,
  PRIMARY KEY ("source", "item_id")
);

DROP TABLE IF EXISTS "print_design_palettes";

CREATE TABLE IF NOT EXISTS "print_design_palettes" (
  "source" TEXT,
  "item_id" TEXT,
  "colour" CHAR(7),
  PRIMARY KEY ("source", "item_id", "colour")
  FOREIGN KEY ("item_id", "source") REFERENCES "clothes" ("item_id", "source")
);

DROP TABLE IF EXISTS "print_design_regions";

CREATE TABLE IF NOT EXISTS "print_design_regions" (
  "source" TEXT,
  "item_id" TEXT,
  "left" INTEGER,
  "top" INTEGER,
  "width" INTEGER,
  "height" INTEGER,
  PRIMARY KEY ("source", "item_id")
  FOREIGN KEY ("item_id", "source") REFERENCES "clothes" ("item_id", "source")
);

DROP TABLE IF EXISTS "palette_distances";

CREATE TABLE IF NOT EXISTS "palette_distances" (
  "palette_id" INTEGER,
  "source" TEXT,
  "design_id" TEXT,
  "distance" FLOAT,
  PRIMARY KEY ("palette_id", "source", "design_id")
  FOREIGN KEY ("palette_id") REFERENCES "palettes" ("id")
  FOREIGN KEY ("design_id", "source") REFERENCES "clothes" ("item_id", "source")
);
