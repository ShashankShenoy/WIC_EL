declare module 'react-plotly.js' {
  import { Component } from 'react';
  import { Config, Data, Layout } from 'plotly.js';

  interface PlotParams {
    data: Data[];
    layout?: Partial<Layout>;
    config?: Partial<Config>;
    frames?: any[];
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (err: any) => void;
    divId?: string;
    revision?: number;
    onClickAnnotation?: (event: any) => void;
    onLegendClick?: (event: any) => boolean;
    onLegendDoubleClick?: (event: any) => boolean;
    onRelayout?: (event: any) => void;
    onRestyle?: (data: any) => void;
    onRedraw?: () => void;
    onSelected?: (event: any) => void;
    onSelecting?: (event: any) => void;
    onDeselect?: () => void;
    onDoubleClick?: () => void;
    onHover?: (event: any) => void;
    onUnhover?: (event: any) => void;
    onAnimationInterrupted?: () => void;
  }

  export default class Plot extends Component<PlotParams> {}
}
