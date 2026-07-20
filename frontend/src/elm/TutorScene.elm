port module TutorScene exposing (main)

import Browser
import Html exposing (Html)
import Html.Attributes as HtmlAttributes
import Json.Decode as Decode exposing (Decoder, Value)
import Json.Encode as Encode
import Svg exposing (Svg)
import Svg.Attributes as SvgAttributes
import Svg.Events as SvgEvents


port sceneStateIn : (Value -> msg) -> Sub msg


port interact : Encode.Value -> Cmd msg


type Scene
    = Plot PlotScene
    | Unavailable


type alias Point =
    { x : Float
    , y : Float
    }


type alias Viewport =
    { xMin : Float
    , xMax : Float
    , yMin : Float
    , yMax : Float
    }


type alias PlotScene =
    { viewport : Viewport
    , curveSegments : List (List Point)
    , shadeSegments : List (List Point)
    , marker : Maybe Point
    }


type alias Model =
    { scene : Scene
    , status : String
    }


type Msg
    = ReceiveScene Value
    | SceneInteracted


main : Program Value Model Msg
main =
    Browser.element
        { init = \flags -> ( decodeState flags, Cmd.none )
        , update = update
        , subscriptions = \_ -> sceneStateIn ReceiveScene
        , view = view
        }


update : Msg -> Model -> ( Model, Cmd Msg )
update message model =
    case message of
        ReceiveScene value ->
            ( decodeState value, Cmd.none )

        SceneInteracted ->
            ( model
            , interact
                (Encode.object
                    [ ( "type", Encode.string "scene_click" )
                    ]
                )
            )


decodeState : Value -> Model
decodeState value =
    case Decode.decodeValue stateDecoder value of
        Ok model ->
            model

        Err _ ->
            { scene = Unavailable
            , status = "Rich visual unavailable: the scene data was invalid."
            }


stateDecoder : Decoder Model
stateDecoder =
    Decode.map2
        (\maybeScene status ->
            { scene = Maybe.withDefault Unavailable maybeScene
            , status = status
            }
        )
        (Decode.field "scene" (Decode.nullable sceneDecoder))
        (Decode.field "status" Decode.string)


sceneDecoder : Decoder Scene
sceneDecoder =
    Decode.field "kind" Decode.string
        |> Decode.andThen
            (\kind ->
                case kind of
                    "plot" ->
                        Decode.map Plot plotDecoder

                    _ ->
                        Decode.fail ("unknown scene kind: " ++ kind)
            )


plotDecoder : Decoder PlotScene
plotDecoder =
    Decode.map4
        (\viewport curveSegments shadeSegments marker ->
            { viewport = viewport
            , curveSegments = curveSegments
            , shadeSegments = shadeSegments
            , marker = marker
            }
        )
        (Decode.field "viewport" viewportDecoder)
        (Decode.field "curveSegments" (Decode.list (Decode.list pointDecoder)))
        (Decode.field "shadeSegments" (Decode.list (Decode.list pointDecoder)))
        (Decode.field "marker" (Decode.nullable pointDecoder))
        |> Decode.andThen validatePlot


viewportDecoder : Decoder Viewport
viewportDecoder =
    Decode.map4
        (\xMin xMax yMin yMax ->
            { xMin = xMin
            , xMax = xMax
            , yMin = yMin
            , yMax = yMax
            }
        )
        (Decode.field "xMin" Decode.float)
        (Decode.field "xMax" Decode.float)
        (Decode.field "yMin" Decode.float)
        (Decode.field "yMax" Decode.float)
        |> Decode.andThen
            (\viewport ->
                if
                    finite viewport.xMin
                        && finite viewport.xMax
                        && finite viewport.yMin
                        && finite viewport.yMax
                        && viewport.xMin
                        < viewport.xMax
                        && viewport.yMin
                        < viewport.yMax
                        && viewport.xMin
                        == -5
                        && viewport.xMax
                        == 5
                        && viewport.yMin
                        == -5
                        && viewport.yMax
                        == 5
                then
                    Decode.succeed viewport

                else
                    Decode.fail "the Phase 1 viewport must be [-5, 5] in both axes"
            )


pointDecoder : Decoder Point
pointDecoder =
    Decode.map2 Point
        (Decode.field "x" Decode.float)
        (Decode.field "y" Decode.float)
        |> Decode.andThen
            (\point ->
                if finite point.x && finite point.y then
                    Decode.succeed point

                else
                    Decode.fail "scene points must be finite"
            )


finite : Float -> Bool
finite value =
    not (isNaN value || isInfinite value)


pointInside : Viewport -> Point -> Bool
pointInside viewport point =
    point.x
        >= viewport.xMin
        && point.x
        <= viewport.xMax
        && point.y
        >= viewport.yMin
        && point.y
        <= viewport.yMax


validatePlot : PlotScene -> Decoder PlotScene
validatePlot scene =
    let
        validCurves =
            not (List.isEmpty scene.curveSegments)
                && List.length scene.curveSegments
                <= 128
                && List.all
                    (\segment ->
                        List.length segment >= 2
                            && List.length segment
                            <= 10000
                            && List.all (pointInside scene.viewport) segment
                    )
                    scene.curveSegments

        validShades =
            List.length scene.shadeSegments
                <= 128
                && List.all
                (\segment ->
                    List.length segment >= 3
                        && List.length segment
                        <= 10000
                        && List.all (pointInside scene.viewport) segment
                )
                scene.shadeSegments

        validMarker =
            Maybe.map (pointInside scene.viewport) scene.marker
                |> Maybe.withDefault True
    in
    if validCurves && validShades && validMarker then
        Decode.succeed scene

    else
        Decode.fail "plot geometry is outside the supported viewport"


canvasWidth : Float
canvasWidth =
    600


canvasHeight : Float
canvasHeight =
    320


screenX : Viewport -> Float -> Float
screenX viewport x =
    (x - viewport.xMin) / (viewport.xMax - viewport.xMin) * canvasWidth


screenY : Viewport -> Float -> Float
screenY viewport y =
    canvasHeight - (y - viewport.yMin) / (viewport.yMax - viewport.yMin) * canvasHeight


screenPoint : Viewport -> Point -> String
screenPoint viewport point =
    String.fromFloat (screenX viewport point.x)
        ++ ","
        ++ String.fromFloat (screenY viewport point.y)


view : Model -> Html Msg
view model =
    Html.div
        [ HtmlAttributes.class "scene-frame"
        , HtmlAttributes.attribute "data-scene-status" model.status
        ]
        (case model.scene of
            Plot scene ->
                [ viewPlot scene ]
                    ++ viewStatus model.status

            Unavailable ->
                [ Html.div
                    [ HtmlAttributes.class "scene-placeholder"
                    , HtmlAttributes.attribute "role" "status"
                    ]
                    [ Html.text model.status ]
                ]
        )


viewStatus : String -> List (Html Msg)
viewStatus status =
    if String.isEmpty status then
        []

    else
        [ Html.div
            [ HtmlAttributes.class "scene-status"
            , HtmlAttributes.attribute "role" "status"
            ]
            [ Html.text status ]
        ]


viewPlot : PlotScene -> Svg Msg
viewPlot scene =
    Svg.svg
        [ SvgAttributes.viewBox "0 0 600 320"
        , HtmlAttributes.attribute "role" "img"
        , HtmlAttributes.attribute "aria-label" "Interactive coordinate-plane graph"
        , SvgEvents.onClick SceneInteracted
        ]
        ([ Svg.rect
            [ SvgAttributes.x "0"
            , SvgAttributes.y "0"
            , SvgAttributes.width (String.fromFloat canvasWidth)
            , SvgAttributes.height (String.fromFloat canvasHeight)
            , SvgAttributes.class "scene-background"
            ]
            []
         ]
            ++ viewGrid scene.viewport
            ++ viewAxes scene.viewport
            ++ List.map (viewShade scene.viewport) scene.shadeSegments
            ++ List.map (viewCurve scene.viewport) scene.curveSegments
            ++ viewMarker scene.viewport scene.marker
        )


viewGrid : Viewport -> List (Svg Msg)
viewGrid viewport =
    let
        vertical =
            List.range (ceiling viewport.xMin) (floor viewport.xMax)
                |> List.map
                    (\x ->
                        Svg.line
                            [ SvgAttributes.x1 (String.fromFloat (screenX viewport (toFloat x)))
                            , SvgAttributes.y1 "0"
                            , SvgAttributes.x2 (String.fromFloat (screenX viewport (toFloat x)))
                            , SvgAttributes.y2 (String.fromFloat canvasHeight)
                            , SvgAttributes.class "scene-grid-line"
                            ]
                            []
                    )

        horizontal =
            List.range (ceiling viewport.yMin) (floor viewport.yMax)
                |> List.map
                    (\y ->
                        Svg.line
                            [ SvgAttributes.x1 "0"
                            , SvgAttributes.y1 (String.fromFloat (screenY viewport (toFloat y)))
                            , SvgAttributes.x2 (String.fromFloat canvasWidth)
                            , SvgAttributes.y2 (String.fromFloat (screenY viewport (toFloat y)))
                            , SvgAttributes.class "scene-grid-line"
                            ]
                            []
                    )
    in
    vertical ++ horizontal


viewAxes : Viewport -> List (Svg Msg)
viewAxes viewport =
    [ Svg.line
        [ SvgAttributes.x1 (String.fromFloat (screenX viewport 0))
        , SvgAttributes.y1 "0"
        , SvgAttributes.x2 (String.fromFloat (screenX viewport 0))
        , SvgAttributes.y2 (String.fromFloat canvasHeight)
        , SvgAttributes.class "scene-axis"
        ]
        []
    , Svg.line
        [ SvgAttributes.x1 "0"
        , SvgAttributes.y1 (String.fromFloat (screenY viewport 0))
        , SvgAttributes.x2 (String.fromFloat canvasWidth)
        , SvgAttributes.y2 (String.fromFloat (screenY viewport 0))
        , SvgAttributes.class "scene-axis"
        ]
        []
    ]


viewCurve : Viewport -> List Point -> Svg Msg
viewCurve viewport points =
    Svg.polyline
        [ SvgAttributes.points (String.join " " (List.map (screenPoint viewport) points))
        , SvgAttributes.class "scene-curve"
        , SvgAttributes.fill "none"
        ]
        []


viewShade : Viewport -> List Point -> Svg Msg
viewShade viewport points =
    Svg.polygon
        [ SvgAttributes.points (String.join " " (List.map (screenPoint viewport) points))
        , SvgAttributes.class "scene-shade"
        ]
        []


viewMarker : Viewport -> Maybe Point -> List (Svg Msg)
viewMarker viewport marker =
    case marker of
        Just point ->
            [ Svg.circle
                [ SvgAttributes.cx (String.fromFloat (screenX viewport point.x))
                , SvgAttributes.cy (String.fromFloat (screenY viewport point.y))
                , SvgAttributes.r "6"
                , SvgAttributes.class "scene-marker-halo"
                ]
                []
            , Svg.circle
                [ SvgAttributes.cx (String.fromFloat (screenX viewport point.x))
                , SvgAttributes.cy (String.fromFloat (screenY viewport point.y))
                , SvgAttributes.r "3"
                , SvgAttributes.class "scene-marker"
                ]
                []
            ]

        Nothing ->
            []
